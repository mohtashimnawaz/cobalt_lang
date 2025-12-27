use logos::Logos;

use crate::ast::Span;

#[derive(Logos, Debug, PartialEq, Clone)]
pub enum Token {
    // Keywords
    #[token("let")]
    Let,
    #[token("fn")]
    Fn,
    #[token("if")]
    If,
    #[token("then")]
    Then,
    #[token("else")]
    Else,
    #[token("in")]
    In,

    // Symbols
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,

    #[token("==")]
    EqEq,
    #[token("!=")]
    Ne,
    #[token("<=")]
    Le,
    #[token(">=")]
    Ge,
    #[token("<")]
    Lt,
    #[token(">")]
    Gt,
    #[token("&&")]
    And,
    #[token("||")]
    Or,

    #[token("=")]
    Assign,
    #[token(";")]
    Semi,
    #[token(",")]
    Comma,
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("->")]
    Arrow,
    #[token(":")]
    Colon,

    // Booleans (keywords)
    #[token("true")]
    True,
    #[token("false")]
    False,

    // Identifiers
    #[token("as")]
    As,

    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice().to_string())]
    Ident(String),

    // Float literal (simple form, digits.digits)
    #[regex(r"[0-9]+\.[0-9]+", |lex| lex.slice().parse::<f64>().ok())]
    Float(f64),

    // NOTE: logos requires that callback types match; use `.ok()` above to return Option and handle push filtering later

    // Integer literal
    #[regex(r"[0-9]+", |lex| lex.slice().parse::<i32>().ok())]
    Int(i32),

    // Whitespace (skipped)
    #[regex(r"[ \t\n\r]+", logos::skip)]
    Whitespace,
}

/// Token with span range into the source text
pub type SpannedToken = (Token, Span);

/// Convert source text to a Vec of (Token, Span)
pub fn tokenize(src: &str) -> Vec<SpannedToken> {
    let mut out = Vec::new();
    let mut lexer = Token::lexer(src);
    while let Some(res) = lexer.next() {
        match res {
            Ok(tok) => {
                // skip any whitespace token explicitly (logos::skip should already handle this, but be defensive)
                if let Token::Whitespace = tok {
                    continue;
                }
                let span = lexer.span();
                // Only push tokens that successfully parsed their inner data. Logos returns variants even on parse failure when using `.ok()` callbacks.
                match tok {
                    Token::Int(_) | Token::Float(_) | Token::Ident(_) | Token::Let | Token::Fn | Token::If | Token::Then | Token::Else
                    | Token::In | Token::Plus | Token::Minus | Token::Star | Token::Slash | Token::EqEq | Token::Ne | Token::Le | Token::Ge
                    | Token::Lt | Token::Gt | Token::And | Token::Or | Token::Assign | Token::Semi | Token::Comma | Token::LParen | Token::RParen
                    | Token::LBrace | Token::RBrace | Token::Arrow | Token::Colon | Token::True | Token::False => {
                        out.push((tok, Span { start: span.start, end: span.end }));
                    }
                    _ => {
                        // skip invalid tokens
                        continue;
                    }
                }
            }
            Err(_) => {
                // For now skip lexing errors; later we will report them with spans
                continue;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_simple_let() {
        let src = "let x = 5;";
        let toks = tokenize(src);
        let kinds: Vec<_> = toks.iter().map(|(t, _s)| t.clone()).collect();
        assert_eq!(kinds, vec![
            Token::Let,
            Token::Ident("x".to_string()),
            Token::Assign,
            Token::Int(5),
            Token::Semi,
        ]);

        // verify spans map back to substrings
        let substrs: Vec<_> = toks.iter().map(|(_, s)| &src[s.start..s.end]).collect();
        assert_eq!(substrs, &["let", "x", "=", "5", ";"]);
    }

    #[test]
    fn tokenize_if_cmp() {
        let src = "if x == 0 then y else z";
        let toks = tokenize(src);
        let kinds: Vec<_> = toks.iter().map(|(t, _s)| t.clone()).collect();
        assert_eq!(kinds, vec![
            Token::If,
            Token::Ident("x".to_string()),
            Token::EqEq,
            Token::Int(0),
            Token::Then,
            Token::Ident("y".to_string()),
            Token::Else,
            Token::Ident("z".to_string()),
        ]);
    }
}
