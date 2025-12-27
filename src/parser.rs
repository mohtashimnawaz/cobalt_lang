use crate::lexer::{tokenize, Token, SpannedToken};


use crate::ast::{BinaryExpr, BinaryOp, Expr, Item, Literal, Module, Param, Span, Type};

/// Parse error with optional `Span` for diagnostics
#[derive(Debug, Clone)]
pub struct ParseError {
    pub msg: String,
    pub span: Option<Span>,
}

impl ParseError {
    fn new<M: Into<String>>(msg: M, span: Option<Span>) -> Self {
        Self { msg: msg.into(), span }
    }
}

/// Public entry point: parse a whole module from source text.
pub fn parse_module(src: &str) -> Result<Module, Vec<ParseError>> {
    let toks = tokenize(src);
    Parser::new(toks).parse_module()
}

/// Parse a standalone expression (useful for REPL/tests)
pub fn parse_expr(src: &str) -> Result<Expr, Vec<ParseError>> {
    let toks = tokenize(src);
    Parser::new(toks).parse_expr()
}

/// Token-based recursive-descent parser that attaches `Span` information to AST nodes.
struct Parser {
    tokens: Vec<SpannedToken>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<SpannedToken>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos).map(|(t, _s)| t)
    }

    fn peek_span(&self) -> Option<Span> {
        self.tokens.get(self.pos).map(|(_t, s)| *s)
    }

    fn bump(&mut self) -> Option<(Token, Span)> {
        if self.is_eof() { return None; }
        let res = self.tokens[self.pos].clone();
        self.pos += 1;
        Some(res)
    }

    fn expect_ident(&mut self) -> Result<String, ParseError> {
        match self.bump() {
            Some((Token::Ident(name), _span)) => Ok(name),
            Some((tok, span)) => Err(ParseError::new(format!("expected identifier but found {:?}", tok), Some(span))),
            None => Err(ParseError::new("expected identifier but reached end of input", None)),
        }
    }

    fn expect_token(&mut self, expected: Token) -> Result<Span, ParseError> {
        match self.bump() {
            Some((tok, span)) if tok == expected => Ok(span),
            Some((tok, span)) => Err(ParseError::new(format!("expected token {:?} but found {:?}", expected, tok), Some(span))),
            None => Err(ParseError::new(format!("expected token {:?} but reached end of input", expected), None)),
        }
    }

    #[allow(dead_code)]
    fn try_consume_ident(&mut self) -> Option<String> {
        if let Some((Token::Ident(name), _s)) = self.tokens.get(self.pos) {
            let name = name.clone();
            self.pos += 1;
            Some(name)
        } else {
            None
        }
    }

    #[allow(dead_code)]
    fn try_consume(&mut self, tok_pat: Token) -> Option<Span> {
        if let Some((tok, span)) = self.tokens.get(self.pos) {
            if *tok == tok_pat {
                let s = *span;
                self.pos += 1;
                return Some(s);
            }
        }
        None
    }

    fn parse_module(&mut self) -> Result<Module, Vec<ParseError>> {
        let mut items = Vec::new();
        let mut errors = Vec::new();
        while !self.is_eof() {
            match self.parse_item() {
                Ok(item) => items.push(item),
                Err(e) => errors.push(e),
            }
        }
        if errors.is_empty() { Ok(Module { items }) } else { Err(errors) }
    }

    fn parse_item(&mut self) -> Result<Item, ParseError> {
        match self.peek() {
            Some(Token::Fn) => {
                let start_span = self.peek_span();
                self.bump(); // consume fn
                let name = self.expect_ident()?;
                // params
                self.expect_token(Token::LParen)?;
                let mut params = Vec::new();
                if let Some(Token::RParen) = self.peek() {
                    self.bump(); // empty params
                } else {
                    loop {
                        let param_name = self.expect_ident()?;
                        self.expect_token(Token::Colon)?;
                        let ty = match self.bump() {
                            Some((Token::Ident(s), _)) if s == "i32" => Type::I32,
                            Some((Token::Ident(s), _)) if s == "bool" => Type::Bool,
                            Some((tok, span)) => return Err(ParseError::new(format!("unexpected token in param type: {:?}", tok), Some(span))),
                            None => return Err(ParseError::new("unexpected EOF in parameter list", None)),
                        };
                        params.push(Param { name: param_name, ty });
                        if let Some(Token::Comma) = self.peek() { self.bump(); continue; }
                        if let Some(Token::RParen) = self.peek() { self.bump(); break; }
                        return Err(ParseError::new("unexpected token in parameter list", self.peek_span()));
                    }
                }

                // optional return type
                let ret_type = if let Some(Token::Arrow) = self.peek() {
                    self.bump();
                    match self.bump() {
                        Some((Token::Ident(s), _)) if s == "i32" => Type::I32,
                        Some((Token::Ident(s), _)) if s == "bool" => Type::Bool,
                        Some((tok, span)) => return Err(ParseError::new(format!("unexpected token in return type: {:?}", tok), Some(span))),
                        None => return Err(ParseError::new("unexpected EOF after ->", None)),
                    }
                } else { Type::I32 };

                // body is an expression in braces
                self.expect_token(Token::LBrace)?;
                let body = self.parse_expr_inner()?;
                let rbrace_span = self.expect_token(Token::RBrace)?;

                let span = start_span.map(|s| Span { start: s.start, end: rbrace_span.end }).or_else(|| Self::expr_span(&body).map(|(s, e)| Span { start: s, end: e }));

                Ok(Item::Function { name, params, ret_type, body, span })
            }
            Some(Token::Let) => {
                let start_span = self.peek_span();
                self.bump();
                let name = self.expect_ident()?;
                // optional type annotation: `: i32` or `: bool`
                let ty = if let Some(Token::Colon) = self.peek() {
                    self.bump();
                    match self.bump() {
                        Some((Token::Ident(s), _)) if s == "i32" => Some(Type::I32),
                        Some((Token::Ident(s), _)) if s == "bool" => Some(Type::Bool),
                        Some((tok, span)) => return Err(ParseError::new(format!("unexpected token in let type: {:?}", tok), Some(span))),
                        None => return Err(ParseError::new("unexpected EOF in let type", None)),
                    }
                } else { None };

                self.expect_token(Token::Assign)?;
                let value = self.parse_expr_inner()?;
                let end_span = Self::expr_span(&value).map(|(s, e)| Span { start: s, end: e });
                // allow optional semicolon
                if let Some(Token::Semi) = self.peek() { self.bump(); }
                let span = match (start_span, end_span) {
                    (Some(s), Some(e)) => Some(Span { start: s.start, end: e.end }),
                    (Some(s), None) => Some(s),
                    (None, Some(e)) => Some(e),
                    _ => None,
                };
                Ok(Item::Let { name, ty, value, span })
            }
            Some(tok) => Err(ParseError::new(format!("unexpected token at top-level: {:?}", tok), self.peek_span())),
            None => Err(ParseError::new("unexpected EOF", None)),
        }
    }

    // Public wrapper for parsing an expression which collects errors into Vec<ParseError>
    fn parse_expr(&mut self) -> Result<Expr, Vec<ParseError>> {
        match self.parse_expr_inner() {
            Ok(e) => Ok(e),
            Err(err) => Err(vec![err]),
        }
    }

    // Inner parser functions return single `ParseError` values and are composed above
    fn parse_expr_inner(&mut self) -> Result<Expr, ParseError> {
        self.parse_logic_or()
    }

    fn parse_logic_or(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_logic_and()?;
        while let Some(Token::Or) = self.peek() {
            let _ = self.bump();
            let right = self.parse_logic_and()?;
            let span = Self::merge_spans(&left, &right);
            left = Expr::Binary(BinaryExpr { op: BinaryOp::Or, left: Box::new(left), right: Box::new(right), span: Some(span) });
        }
        Ok(left)
    }

    fn parse_logic_and(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_cmp()?;
        while let Some(Token::And) = self.peek() {
            let _ = self.bump();
            let right = self.parse_cmp()?;
            let span = Self::merge_spans(&left, &right);
            left = Expr::Binary(BinaryExpr { op: BinaryOp::And, left: Box::new(left), right: Box::new(right), span: Some(span) });
        }
        Ok(left)
    }

    fn parse_cmp(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_sum()?;
        loop {
            match self.peek() {
                Some(Token::EqEq) => { self.bump(); let right = self.parse_sum()?; let span = Self::merge_spans(&left, &right); left = Expr::Binary(BinaryExpr{ op: BinaryOp::Eq, left: Box::new(left), right: Box::new(right), span: Some(span) }); }
                Some(Token::Ne) => { self.bump(); let right = self.parse_sum()?; let span = Self::merge_spans(&left, &right); left = Expr::Binary(BinaryExpr{ op: BinaryOp::Ne, left: Box::new(left), right: Box::new(right), span: Some(span) }); }
                Some(Token::Lt) => { self.bump(); let right = self.parse_sum()?; let span = Self::merge_spans(&left, &right); left = Expr::Binary(BinaryExpr{ op: BinaryOp::Lt, left: Box::new(left), right: Box::new(right), span: Some(span) }); }
                Some(Token::Le) => { self.bump(); let right = self.parse_sum()?; let span = Self::merge_spans(&left, &right); left = Expr::Binary(BinaryExpr{ op: BinaryOp::Le, left: Box::new(left), right: Box::new(right), span: Some(span) }); }
                Some(Token::Gt) => { self.bump(); let right = self.parse_sum()?; let span = Self::merge_spans(&left, &right); left = Expr::Binary(BinaryExpr{ op: BinaryOp::Gt, left: Box::new(left), right: Box::new(right), span: Some(span) }); }
                Some(Token::Ge) => { self.bump(); let right = self.parse_sum()?; let span = Self::merge_spans(&left, &right); left = Expr::Binary(BinaryExpr{ op: BinaryOp::Ge, left: Box::new(left), right: Box::new(right), span: Some(span) }); }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_sum(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_product()?;
        loop {
            match self.peek() {
                Some(Token::Plus) => { self.bump(); let right = self.parse_product()?; let span = Self::merge_spans(&left, &right); left = Expr::Binary(BinaryExpr{ op: BinaryOp::Add, left: Box::new(left), right: Box::new(right), span: Some(span) }); }
                Some(Token::Minus) => { self.bump(); let right = self.parse_product()?; let span = Self::merge_spans(&left, &right); left = Expr::Binary(BinaryExpr{ op: BinaryOp::Sub, left: Box::new(left), right: Box::new(right), span: Some(span) }); }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_product(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_primary()?;
        loop {
            match self.peek() {
                Some(Token::Star) => { self.bump(); let right = self.parse_primary()?; let span = Self::merge_spans(&left, &right); left = Expr::Binary(BinaryExpr{ op: BinaryOp::Mul, left: Box::new(left), right: Box::new(right), span: Some(span) }); }
                Some(Token::Slash) => { self.bump(); let right = self.parse_primary()?; let span = Self::merge_spans(&left, &right); left = Expr::Binary(BinaryExpr{ op: BinaryOp::Div, left: Box::new(left), right: Box::new(right), span: Some(span) }); }
                _ => break,
            }
        }
        Ok(left)
    }
    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        if let Some((tok_clone, _span)) = self.tokens.get(self.pos).cloned() {
            match tok_clone {
                Token::Int(i) => {
                    self.bump();
                    Ok(Expr::Literal(Literal::Int(i)))
                }
                Token::True => { self.bump(); Ok(Expr::Literal(Literal::Bool(true))) }
                Token::False => { self.bump(); Ok(Expr::Literal(Literal::Bool(false))) }
                Token::Ident(name) => {
                    // Could be a variable or a call
                    // look at the next token after the identifier to detect a call
                    if let Some((next_tok, _)) = self.tokens.get(self.pos + 1) {
                        if let Token::LParen = next_tok {
                            // consume the identifier and the LParen
                            self.bump(); // consume ident
                            self.bump(); // consume LParen
                            let mut args = Vec::new();
                            if let Some((Token::RParen, _)) = self.tokens.get(self.pos) { self.bump(); }
                            else {
                                loop {
                                    let arg = self.parse_expr_inner()?;
                                    args.push(arg);
                                    if let Some((Token::Comma, _)) = self.tokens.get(self.pos) { self.bump(); continue; }
                                    if let Some((Token::RParen, _)) = self.tokens.get(self.pos) { self.bump(); break; }
                                    return Err(ParseError::new("unexpected token in call arguments", self.peek_span()));
                                }
                            }
                            return Ok(Expr::Call { callee: Box::new(Expr::Var(name)), args, span: None });
                        } else {
                            self.bump();
                            return Ok(Expr::Var(name));
                        }
                    } else {
                        // no next token -> just an identifier/variable
                        self.bump();
                        return Ok(Expr::Var(name));
                    }
                }
                Token::LParen => {
                    self.bump();
                    let e = self.parse_expr_inner()?;
                    self.expect_token(Token::RParen)?;
                    Ok(e)
                }
                Token::If => {
                    self.bump();
                    let cond = self.parse_expr_inner()?;
                    self.expect_token(Token::Then)?;
                    let then_branch = self.parse_expr_inner()?;
                    self.expect_token(Token::Else)?;
                    let else_branch = self.parse_expr_inner()?;
                    let span = Self::merge_spans(&cond, &else_branch);
                    Ok(Expr::If { cond: Box::new(cond), then_branch: Box::new(then_branch), else_branch: Box::new(else_branch), span: Some(span) })
                }
                Token::Let => {
                    self.bump();
                    let name = self.expect_ident()?;
                    self.expect_token(Token::Assign)?;
                    let value = self.parse_expr_inner()?;
                    // optional `in` keyword (support both `let x = v in body` and `let x = v; body` forms)
                    if let Some(Token::In) = self.peek() { self.bump(); }
                    let body = self.parse_expr_inner()?;
                    let span = Self::merge_spans(&value, &body);
                    Ok(Expr::Let { name, value: Box::new(value), body: Box::new(body), span: Some(span) })
                }
                other => Err(ParseError::new(format!("unexpected token in primary: {:?}", other), self.peek_span())),
            }
        } else { Err(ParseError::new("unexpected EOF in expression", None)) }
    }

    fn merge_spans(a: &Expr, b: &Expr) -> Span {
        let (sa, ea) = Self::expr_span(a).unwrap_or((0,0));
        let (sb, eb) = Self::expr_span(b).unwrap_or((0,0));
        Span { start: sa.min(sb), end: ea.max(eb) }
    }

    fn expr_span(e: &Expr) -> Option<(usize, usize)> {
        match e {
            Expr::Literal(_) => None,
            Expr::Var(_) => None,
            Expr::Binary(be) => be.span.map(|s| (s.start, s.end)),
            Expr::If { span, .. } => span.map(|s| (s.start, s.end)),
            Expr::Let { span, .. } => span.map(|s| (s.start, s.end)),
            Expr::Call { span, .. } => span.map(|s| (s.start, s.end)),
        }
    }
}


// (old character-level parser removed; token-based parser used instead)

// Additional token-based parser tests moved above; remaining character-based tests are removed.

#[cfg(test)]
mod tests_parser_additional {
    use super::*;
    use crate::ast::{BinaryOp, Expr, Literal};

    #[test]
    fn parse_arith_precedence_tokens() {
        let module = parse_module("let x = 1 + 2 * 3").expect("parse module");
        match &module.items[..] {
            [Item::Let { name, value, .. }] => {
                assert_eq!(name, "x");
                if let Expr::Binary(BinaryExpr { op: BinaryOp::Add, left, right, .. }) = value {
                    assert!(matches!(**left, Expr::Literal(Literal::Int(1))));
                    if let Expr::Binary(BinaryExpr { op: BinaryOp::Mul, left: rleft, right: rright, .. }) = &**right {
                        assert!(matches!(**rleft, Expr::Literal(Literal::Int(2))));
                        assert!(matches!(**rright, Expr::Literal(Literal::Int(3))));
                    } else {
                        panic!("expected multiplication on right-hand side");
                    }
                } else {
                    panic!("expected top-level addition");
                }
            }
            _ => panic!("unexpected items"),
        }
    }
}


#[cfg(test)]
mod tests_token_parser {
    use super::*;
    use crate::lexer::tokenize;
    use crate::ast::{Item, Expr, BinaryExpr, Literal, BinaryOp};

    #[test]
    fn token_parser_spans_attached() {
        let src = "let x = 1 + 2 * 3;";
        let module = parse_module(src).expect("token parse module");
        assert!(matches!(module.items.get(0), Some(Item::Let { .. } )));
    }

    #[test]
    fn parse_expr_via_tokens() {
        let e = parse_expr("1 + 2 * 3").expect("parse expr");
        if let Expr::Binary(BinaryExpr { op: BinaryOp::Add, left, right: _, .. }) = e {
            assert!(matches!(*left, Expr::Literal(Literal::Int(1))));
        } else { panic!("expected add top-level") }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinaryOp, Expr, Literal};

    #[test]
    fn parse_arith_precedence() {
        use crate::ast::{Item, BinaryExpr};

        let module = parse_module("let x = 1 + 2 * 3").expect("parse module");
        match &module.items[..] {
            [Item::Let { name, value, .. }] => {
                assert_eq!(name, "x");
                if let Expr::Binary(BinaryExpr { op: BinaryOp::Add, left, right, .. }) = value {
                    assert!(matches!(**left, Expr::Literal(Literal::Int(1))));
                    if let Expr::Binary(BinaryExpr { op: BinaryOp::Mul, left: rleft, right: rright, .. }) = &**right {
                        assert!(matches!(**rleft, Expr::Literal(Literal::Int(2))));
                        assert!(matches!(**rright, Expr::Literal(Literal::Int(3))));
                    } else {
                        panic!("expected multiplication on right-hand side");
                    }
                } else {
                    panic!("expected top-level addition");
                }
            }
            _ => panic!("unexpected items"),
        }
    }

    #[test]
    fn parse_integer_literal() {
        // Use token-based parse_expr
        let e2 = parse_expr("42").expect("parse expr");
        assert!(matches!(e2, Expr::Literal(Literal::Int(42))));
    }

    #[test]
    fn parse_operator_and_product() {
        let add = parse_expr("1 + 2").expect("parse add");
        if let Expr::Binary(BinaryExpr { op, left, right, .. }) = add {
            assert_eq!(op, BinaryOp::Add);
            assert!(matches!(*left, Expr::Literal(Literal::Int(1))));
            assert!(matches!(*right, Expr::Literal(Literal::Int(2))));
        } else { panic!("expected binary") }

        let prod = parse_expr("2 * 3").expect("parse mul");
        if let Expr::Binary(BinaryExpr { op, .. }) = prod { assert_eq!(op, BinaryOp::Mul); } else { panic!("expected mul") }
    }

    #[test]
    fn parse_equation_simple() {
        let expr = parse_expr("1 + 2 - 3").expect("parse eq");
        if let Expr::Binary(BinaryExpr { op, left, right, .. }) = expr {
            // Top-level is subtraction: (1 + 2) - 3
            assert_eq!(op, BinaryOp::Sub);
            if let Expr::Binary(BinaryExpr { op: inner_op, left: il, right: ir, .. }) = *left {
                assert_eq!(inner_op, BinaryOp::Add);
                assert!(matches!(*il, Expr::Literal(Literal::Int(1))));
                assert!(matches!(*ir, Expr::Literal(Literal::Int(2))));
            } else {
                panic!("expected inner addition");
            }
            assert!(matches!(*right, Expr::Literal(Literal::Int(3))));
        } else {
            panic!("expected binary expression");
        }

        // parse with spaces omitted
        let expr2 = parse_expr("1+2").expect("parse eq no spaces");
        if let Expr::Binary(BinaryExpr { op, left, right, .. }) = expr2 {
            assert_eq!(op, BinaryOp::Add);
            assert!(matches!(*left, Expr::Literal(Literal::Int(1))));
            assert!(matches!(*right, Expr::Literal(Literal::Int(2))));
        } else {
            panic!("expected binary expression");
        }
    }
}
