use chumsky::prelude::*;

use crate::ast::{BinaryExpr, BinaryOp, Expr, Item, Literal, Module, Param, Type};

/// Public entry point: parse a whole module from source text.
pub fn parse_module(src: &str) -> Result<Module, Vec<chumsky::error::Simple<char>>> {
    module_parser().parse(src)
}

fn module_parser() -> impl Parser<char, Module, Error = Simple<char>> {

    let ident = text::ident().padded();

    let int_lit = text::int(10)
        .padded()
        .map(|s: String| Expr::Literal(Literal::Int(s.parse().unwrap())));

    let bool_lit = just("true")
        .to(Expr::Literal(Literal::Bool(true)))
        .or(just("false").to(Expr::Literal(Literal::Bool(false))))
        .padded();

    let atom = recursive(|expr| {
        let paren = expr.clone().delimited_by(just('(').padded(), just(')').padded());

        let var = ident.clone().map(|s: String| Expr::Var(s));

        let if_expr = just("if")
            .padded()
            .ignore_then(expr.clone())
            .then_ignore(just("then").padded())
            .then(expr.clone())
            .then_ignore(just("else").padded())
            .then(expr.clone())
            .map(|((cond, then_branch), else_branch)| Expr::If {
                cond: Box::new(cond),
                then_branch: Box::new(then_branch),
                else_branch: Box::new(else_branch),
                span: None,
            });

        let call = var.clone().then(
            expr.clone()
                .separated_by(just(',').padded())
                .delimited_by(just('(').padded(), just(')').padded()),
        )
        .map(|(callee, args)| Expr::Call {
            callee: Box::new(callee),
            args,
            span: None,
        });

        choice((paren, int_lit, bool_lit, if_expr, call, var)).boxed()
    });

    // Binary operator helpers (left-associative folding)
    let product = atom.clone().then(
        (just('*').to(BinaryOp::Mul).or(just('/').to(BinaryOp::Div))).then(atom.clone()).repeated(),
    ).map(|(first, rest)| {
        rest.into_iter().fold(first, |lhs, (op, rhs)| Expr::Binary(BinaryExpr {
            op,
            left: Box::new(lhs),
            right: Box::new(rhs),
            span: None,
        }))
    });

    let sum = product.clone().then(
        (just('+').to(BinaryOp::Add).or(just('-').to(BinaryOp::Sub))).then(product).repeated(),
    ).map(|(first, rest)| {
        rest.into_iter().fold(first, |lhs, (op, rhs)| Expr::Binary(BinaryExpr {
            op,
            left: Box::new(lhs),
            right: Box::new(rhs),
            span: None,
        }))
    });

    let cmp = sum.clone().then(
        (
            just("==").to(BinaryOp::Eq)
            .or(just("!=").to(BinaryOp::Ne))
            .or(just("<=").to(BinaryOp::Le))
            .or(just(">=").to(BinaryOp::Ge))
            .or(just("<").to(BinaryOp::Lt))
            .or(just(">").to(BinaryOp::Gt))
        ).then(sum).repeated(),
    ).map(|(first, rest)| {
        rest.into_iter().fold(first, |lhs, (op, rhs)| Expr::Binary(BinaryExpr {
            op,
            left: Box::new(lhs),
            right: Box::new(rhs),
            span: None,
        }))
    });

    let logic_and = cmp.clone().then((just("&&").to(BinaryOp::And)).then(cmp).repeated()).map(|(first, rest)| {
        rest.into_iter().fold(first, |lhs, (op, rhs)| Expr::Binary(BinaryExpr {
            op,
            left: Box::new(lhs),
            right: Box::new(rhs),
            span: None,
        }))
    });

    let logic_or = logic_and.clone().then((just("||").to(BinaryOp::Or)).then(logic_and).repeated()).map(|(first, rest)| {
        rest.into_iter().fold(first, |lhs, (op, rhs)| Expr::Binary(BinaryExpr {
            op,
            left: Box::new(lhs),
            right: Box::new(rhs),
            span: None,
        }))
    });

    // let-expr: `let <ident> = <expr> in <expr>`
    let let_expr = just("let")
        .padded()
        .ignore_then(ident.clone())
        .then_ignore(just('=').padded())
        .then(logic_or.clone())
        .then_ignore(just("in").padded())
        .then(logic_or.clone())
        .map(|((name, value), body)| Expr::Let {
            name,
            value: Box::new(value),
            body: Box::new(body),
            span: None,
        });

    // An expression can be a let-binding or logical/or expression
    let expr = let_expr.or(logic_or);

    // Type parser (simple)
    let type_parser = just("i32").to(Type::I32).or(just("bool").to(Type::Bool)).padded();

    // Param parser: `ident: i32`
    let param = ident.clone().then_ignore(just(':').padded()).then(type_parser.clone()).map(|(name, ty)| Param { name, ty });

    let fn_item = just("fn")
        .padded()
        .ignore_then(ident.clone())
        .then(param.clone().separated_by(just(',').padded()).delimited_by(just('(').padded(), just(')').padded()))
        .then(just("->").padded().ignore_then(type_parser.clone()).or_not())
        .then(expr.clone().delimited_by(just('{').padded(), just('}').padded()))
        .map(|(((name, params), ret), body)| Item::Function {
            name,
            params,
            ret_type: ret.unwrap_or(Type::I32),
            body,
            span: None,
        });

    let let_item = just("let")
        .padded()
        .ignore_then(ident.clone())
        .then_ignore(just('=').padded())
        .then(expr.clone())
        .map(|(name, value)| Item::Let { name, value, span: None });

    let item = fn_item.or(let_item).padded();

    item.repeated().then_ignore(end()).map(|items| Module { items })
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
}
