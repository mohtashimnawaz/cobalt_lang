//! AST for Cobalt (Phase 1)
//!
//! This module defines the core AST nodes sufficient for the MVP:
//! - i32 and bool types
//! - immutable `let` bindings
//! - `if` expressions
//! - binary math ops (+, -, *, /) and common comparisons
//! - function definitions and calls
//!
//! Design notes:
//! - Binary operations are represented with a `BinaryOp` enum and a `BinaryExpr` containing boxed operands.
//!   This makes it straightforward for the parser to construct nodes and for later lowering / codegen to
//!   pattern-match on each operator.
//! - Recursive expressions use `Box<Expr>` to keep enum size bounded and enable arbitrarily deep nesting.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Type {
    I32,
    Bool,
    /// A function type: (param types...) -> return type
    Func { params: Vec<Type>, ret: Box<Type> },
}

#[derive(Clone, Debug, PartialEq)]
pub enum Literal {
    Int(i32),
    Bool(bool),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,

    /// Comparisons (useful for `if` conditions)
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    /// Logical ops (for booleans)
    And,
    Or,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BinaryExpr {
    pub op: BinaryOp,
    pub left: Box<Expr>,
    pub right: Box<Expr>,
    pub span: Option<Span>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Literal(Literal),
    Var(String),
    Binary(BinaryExpr),

    /// `if cond { then } else { else }` (all are expressions)
    If {
        cond: Box<Expr>,
        then_branch: Box<Expr>,
        else_branch: Box<Expr>,
        span: Option<Span>,
    },

    /// Expression-level immutable binding: `let name = value; body`
    /// This is expression-scoped and evaluates to `body` with `name` bound.
    Let {
        name: String,
        value: Box<Expr>,
        body: Box<Expr>,
        span: Option<Span>,
    },

    /// Function call: `callee(arg1, arg2, ...)`
    Call {
        callee: Box<Expr>,
        args: Vec<Expr>,
        span: Option<Span>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub name: String,
    pub ty: Type,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Item {
    /// Top-level function definition
    Function {
        name: String,
        params: Vec<Param>,
        ret_type: Type,
        body: Expr,
        span: Option<Span>,
    },

    /// Top-level immutable binding: `let name = value` (no body at top level)
    Let {
        name: String,
        value: Expr,
        span: Option<Span>,
    },
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Module {
    pub items: Vec<Item>,
}

impl Module {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }
}
