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
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice().to_string())]
    Ident(String),

    // Integer literal
    #[regex(r"[0-9]+", |lex| lex.slice().parse::<i32>().unwrap())]
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
                out.push((tok, Span { start: span.start, end: span.end }));
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
