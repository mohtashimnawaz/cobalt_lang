use std::collections::HashMap;

use crate::ast::{BinaryExpr, BinaryOp, Expr, Item, Module, Span, Type, Literal};

/// A stack-based symbol table for scoping
#[derive(Debug, Default)]
pub struct SymbolTable {
    scopes: Vec<HashMap<String, Type>>,
}

impl SymbolTable {
    pub fn new() -> Self {
        Self { scopes: vec![HashMap::new()] }
    }

    pub fn push(&mut self) {
        self.scopes.push(HashMap::new());
    }

    pub fn pop(&mut self) {
        self.scopes.pop();
    }

    pub fn insert(&mut self, name: String, ty: Type) -> Option<Type> {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, ty)
        } else {
            None
        }
    }

    pub fn lookup(&self, name: &str) -> Option<Type> {
        for scope in self.scopes.iter().rev() {
            if let Some(t) = scope.get(name) {
                return Some(t.clone());
            }
        }
        None
    }
}

/// Type checking error with optional source `Span` for diagnostics
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeError {
    pub msg: String,
    pub span: Option<Span>,
}

impl TypeError {
    pub fn new<M: Into<String>>(msg: M, span: Option<Span>) -> Self {
        Self { msg: msg.into(), span }
    }
}

/// A simple warning emitted by the type checker (not fatal)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeWarning {
    pub msg: String,
    pub span: Option<Span>,
}

impl TypeWarning {
    pub fn new<M: Into<String>>(msg: M, span: Option<Span>) -> Self {
        Self { msg: msg.into(), span }
    }
}

/// Internal core that returns (errors, warnings)
fn type_check_core(module: &Module) -> (Vec<TypeError>, Vec<TypeWarning>) {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    let mut sym = SymbolTable::new();

    // First pass: collect top-level function signatures and top-level lets
    for item in &module.items {
        match item {
            Item::Function { name, params, ret_type, .. } => {
                let param_types: Vec<Type> = params.iter().map(|p| p.ty.clone()).collect();
                let fty = Type::Func { params: param_types, ret: Box::new(ret_type.clone()) };
                if sym.insert(name.clone(), fty).is_some() {
                    errors.push(TypeError::new(format!("function `{}` redeclared", name), None));
                }
            }
            Item::Let { name, ty, value, span } => {
                // try type-check value; if ok, record its type; if there is an annotated type, check it
                match type_of_expr(value, &sym, &mut warnings) {
                    Ok(inferred) => {
                        if let Some(declared) = ty {
                            if *declared != inferred {
                                errors.push(TypeError::new(format!("top-level let `{}` declared as {:?} but value has type {:?}", name, declared, inferred), span.or_else(|| extract_span_from_expr(value))));
                                // still insert the declared type to be conservative
                                if sym.insert(name.clone(), declared.clone()).is_some() {
                                    warnings.push(TypeWarning::new(format!("top-level let `{}` shadows previous binding", name), span.or_else(|| extract_span_from_expr(value))));
                                }
                            } else {
                                if sym.insert(name.clone(), inferred).is_some() {
                                    warnings.push(TypeWarning::new(format!("top-level let `{}` shadows previous binding", name), span.or_else(|| extract_span_from_expr(value))));
                                }
                            }
                        } else {
                            if sym.insert(name.clone(), inferred).is_some() {
                                warnings.push(TypeWarning::new(format!("top-level let `{}` shadows previous binding", name), span.or_else(|| extract_span_from_expr(value))));
                            }
                        }
                    }
                    Err(mut es) => errors.append(&mut es),
                }
            }
        }
    }

    // Second pass: type-check function bodies (with params in scope)
    for item in &module.items {
        if let Item::Function { name, params, ret_type, body, span } = item {
            sym.push();
            for p in params {
                sym.insert(p.name.clone(), p.ty.clone());
            }
            match type_of_expr(body, &sym, &mut warnings) {
                Ok(body_ty) => {
                    if body_ty != *ret_type {
                        errors.push(TypeError::new(
                            format!("function `{}` declared to return {:?} but body has type {:?}", name, ret_type, body_ty),
                            span.or_else(|| extract_span_from_expr(body)),
                        ));
                    }
                }
                Err(mut es) => errors.append(&mut es),
            }
            sym.pop();
        }
    }

    (errors, warnings)
}

/// Type-check a whole module and return a Vec of `TypeError`s if any (backwards compat)
pub fn type_check_module(module: &Module) -> Result<(), Vec<TypeError>> {
    let (errors, _warnings) = type_check_core(module);
    if errors.is_empty() { Ok(()) } else { Err(errors) }
}

/// Run type-check and also return non-fatal warnings
pub fn type_check_module_with_warnings(module: &Module) -> (Result<(), Vec<TypeError>>, Vec<TypeWarning>) {
    let (errors, warnings) = type_check_core(module);
    if errors.is_empty() { (Ok(()), warnings) } else { (Err(errors), warnings) }
}

fn type_of_expr(expr: &Expr, env: &SymbolTable, warnings: &mut Vec<TypeWarning>) -> Result<Type, Vec<TypeError>> {
    match expr {
        Expr::Literal(lit) => match lit {
            Literal::Int(_) => Ok(Type::I32),
            Literal::Float(_) => Ok(Type::F64),
            Literal::Bool(_) => Ok(Type::Bool),
        },
        Expr::Var(name) => {
            if let Some(ty) = env.lookup(name) {
                Ok(ty)
            } else {
                Err(vec![TypeError::new(format!("unknown variable `{}`", name), None)])
            }
        }
        Expr::Binary(BinaryExpr { op, left, right, span }) => {
            let mut errors = Vec::new();
            let lty = match type_of_expr(left, env, warnings) { Ok(t) => t, Err(mut es) => { errors.append(&mut es); Type::I32 } };
            let rty = match type_of_expr(right, env, warnings) { Ok(t) => t, Err(mut es) => { errors.append(&mut es); Type::I32 } };

            let span_clone = *span;

            let result = match op {
                BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {
                    // numeric promotions: i32 + f64 => f64 (coerce i32 -> f64 with warning)
                    match (&lty, &rty) {
                        (Type::I32, Type::I32) => Ok(Type::I32),
                        (Type::F64, Type::F64) => Ok(Type::F64),
                        (Type::I32, Type::F64) | (Type::F64, Type::I32) => {
                            if let Some(s) = span_clone { warnings.push(TypeWarning::new("promoted i32 to f64 in numeric operation", Some(s))); }
                            Ok(Type::F64)
                        }
                        _ => Err(TypeError::new("arithmetic operands must be numeric and compatible (i32 or f64)", span_clone))
                    }
                }
                BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                    // comparisons: i32/i32 -> bool, f64/f64 -> bool, mixed i32/f64 promote to f64 with warning
                    if lty == rty {
                        Ok(Type::Bool)
                    } else {
                        match (&lty, &rty) {
                            (Type::I32, Type::F64) | (Type::F64, Type::I32) => {
                                if let Some(s) = span_clone { warnings.push(TypeWarning::new("promoted i32 to f64 in comparison", Some(s))); }
                                Ok(Type::Bool)
                            }
                            // allow coercion of integer literal 0/1 to bool and float literal 0.0/1.0 to bool
                            _ => {
                                let _left_is_bool_lit = matches!(**left, Expr::Literal(Literal::Bool(_)));
                                let _right_is_bool_lit = matches!(**right, Expr::Literal(Literal::Bool(_)));
                                let left_is_int01 = matches!(**left, Expr::Literal(Literal::Int(i)) if i == 0 || i == 1);
                                let right_is_int01 = matches!(**right, Expr::Literal(Literal::Int(i)) if i == 0 || i == 1);
                                let left_is_float01 = matches!(**left, Expr::Literal(Literal::Float(f)) if f == 0.0 || f == 1.0);
                                let right_is_float01 = matches!(**right, Expr::Literal(Literal::Float(f)) if f == 0.0 || f == 1.0);

                                if (left_is_int01 && rty == Type::Bool) || (right_is_int01 && lty == Type::Bool) || (left_is_float01 && rty == Type::Bool) || (right_is_float01 && lty == Type::Bool) {
                                    if let Some(s) = span_clone { warnings.push(TypeWarning::new("coerced numeric literal to bool in comparison", Some(s))); }
                                    Ok(Type::Bool)
                                } else {
                                    Err(TypeError::new("comparison operands must have the same type (i32, f64 or bool)", span_clone))
                                }
                            }
                        }
                    }
                }
                BinaryOp::And | BinaryOp::Or => {
                    if lty == Type::Bool && rty == Type::Bool { Ok(Type::Bool) }
                    else { Err(TypeError::new("logical operands must be `bool`", span_clone)) }
                }
            };

            match result {
                Ok(t) => if errors.is_empty() { Ok(t) } else { Err(errors) },
                Err(e) => { errors.push(e); Err(errors) }
            }
        }
        Expr::If { cond, then_branch, else_branch, span } => {
            let mut errors = Vec::new();
            match type_of_expr(cond, env, warnings) {
                Ok(Type::Bool) => {}
                Ok(Type::I32) => {
                    // allow literal 0/1 -> bool coercion with warning
                    if matches!(**cond, Expr::Literal(Literal::Int(i)) if i == 0 || i == 1) {
                        if let Some(s) = *span { warnings.push(TypeWarning::new("coerced integer literal to bool in if condition", Some(s))); }
                    } else {
                        errors.push(TypeError::new(format!("if condition must be `bool`, found i32"), *span));
                    }
                }
                Ok(t) => errors.push(TypeError::new(format!("if condition must be `bool`, found {:?}", t), *span)),
                Err(mut es) => errors.append(&mut es),
            }
            let then_t = match type_of_expr(then_branch, env, warnings) { Ok(t) => t, Err(mut es) => { errors.append(&mut es); Type::I32 } };
            let else_t = match type_of_expr(else_branch, env, warnings) { Ok(t) => t, Err(mut es) => { errors.append(&mut es); Type::I32 } };
            if then_t != else_t { errors.push(TypeError::new("then and else branch types must match", *span)); }
            if errors.is_empty() { Ok(then_t) } else { Err(errors) }
        }
        Expr::Let { name, value, body, span: _ } => {
            let mut errors = Vec::new();
            let vty = match type_of_expr(value, env, warnings) { Ok(t) => t, Err(mut es) => { errors.append(&mut es); Type::I32 } };
            // shadowing allowed; check body in new scope
            let mut env2 = env.clone();
            env2.insert(name.clone(), vty.clone());
            match type_of_expr(body, &env2, warnings) {
                Ok(bt) => if errors.is_empty() { Ok(bt) } else { Err(errors) },
                Err(mut es) => { errors.append(&mut es); Err(errors) }
            }
        }
        Expr::Call { callee, args, span } => {
            // only support direct variable callee
            if let Expr::Var(fn_name) = &**callee {
                if let Some(Type::Func { params, ret }) = env.lookup(fn_name) {
                    let mut errors = Vec::new();
                    if params.len() != args.len() {
                        errors.push(TypeError::new(format!("function `{}` expects {} args but {} were supplied", fn_name, params.len(), args.len()), *span));
                        return Err(errors);
                    }
                    for (i, (arg, expected)) in args.iter().zip(params.iter()).enumerate() {
                        match type_of_expr(arg, env, warnings) {
                            Ok(t) => if &t != expected { errors.push(TypeError::new(format!("argument {} to `{}` expected {:?} but found {:?}", i, fn_name, expected, t), extract_span_from_expr(arg))); }
                            Err(mut es) => errors.append(&mut es),
                        }
                    }
                    if errors.is_empty() { Ok(*ret.clone()) } else { Err(errors) }
                } else {
                    Err(vec![TypeError::new(format!("unknown function `{}`", fn_name), *span)])
                }
            } else {
                Err(vec![TypeError::new("call to non-function expression not supported", *span)])
            }
        }
    }
}

fn extract_span_from_expr(e: &Expr) -> Option<Span> {
    match e {
        Expr::Literal(_) => None,
        Expr::Var(_) => None,
        Expr::Binary(be) => be.span,
        Expr::If { span, .. } => *span,
        Expr::Let { span, .. } => *span,
        Expr::Call { span, .. } => *span,
    }
}

// Make SymbolTable cloneable via deriving from internal scopes
impl Clone for SymbolTable {
    fn clone(&self) -> Self {
        Self { scopes: self.scopes.clone() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_module;

    #[test]
    fn check_simple_add_fn() {
        let src = "fn add(a: i32, b: i32) -> i32 { a + b }";
        let module = parse_module(src).expect("parse module");
        assert!(type_check_module(&module).is_ok());
    }

    #[test]
    fn arithmetic_promotion_to_f64() {
        let src = "fn f() -> f64 { 1 + 2.5 }";
        let module = parse_module(src).expect("parse module");
        let (res, warnings) = type_check_module_with_warnings(&module);
        assert!(res.is_ok());
        assert!(warnings.iter().any(|w| w.msg.contains("promoted i32 to f64")));
    }

    #[test]
    fn error_incompatible_arith() {
        let src = "fn bad() -> i32 { true + 1 }";
        let module = parse_module(src).expect("parse module");
        let err = type_check_module(&module).unwrap_err();
        assert!(err.iter().any(|e| e.msg.contains("arithmetic operands")));
    }

    #[test]
    fn if_coercion_allows_literal() {
        let src = "fn ok() -> i32 { if 1 then 2 else 3 }";
        let module = parse_module(src).expect("parse module");
        let (res, warnings) = type_check_module_with_warnings(&module);
        assert!(res.is_ok());
        assert!(warnings.iter().any(|w| w.msg.contains("coerced integer literal to bool")));
    }

    #[test]
    fn if_cond_var_still_errors() {
        let src = "fn bad(x: i32) -> i32 { if x then 2 else 3 }";
        let module = parse_module(src).expect("parse module");
        let err = type_check_module(&module).unwrap_err();
        assert!(err.iter().any(|e| e.msg.contains("if condition must be `bool`")));
    }

    #[test]
    fn eq_bool_ok() {
        let src = "fn b() -> bool { true == false }";
        let module = parse_module(src).expect("parse module");
        assert!(type_check_module(&module).is_ok());
    }

    #[test]
    fn eq_coercion_allows_int_bool_literal() {
        let src = "fn ok() -> bool { 1 == true }";
        let module = parse_module(src).expect("parse module");
        let (res, warnings) = type_check_module_with_warnings(&module);
        assert!(res.is_ok());
        assert!(warnings.iter().any(|w| w.msg.contains("coerced numeric literal to bool")));
    }

    #[test]
    fn eq_nonliteral_mismatch_errors() {
        let src = "fn bad(x: i32) -> bool { x == true }";
        let module = parse_module(src).expect("parse module");
        let err = type_check_module(&module).unwrap_err();
        assert!(err.iter().any(|e| e.msg.contains("comparison operands must have the same type")));
    }

    #[test]
    fn top_level_let_inferred() {
        let src = "let x = 1 fn f() -> i32 { x }";
        let module = parse_module(src).expect("parse module");
        assert!(type_check_module(&module).is_ok());
    }

    #[test]
    fn top_level_let_annot_mismatch() {
        let src = "let x: i32 = true";
        let module = parse_module(src).expect("parse module");
        let err = type_check_module(&module).unwrap_err();
        assert!(err.iter().any(|e| e.msg.contains("top-level let `x` declared as")));
    }

    #[test]
    fn shadowing_and_nested_scopes() {
        let src = "let x = 1 fn f() -> i32 { let x = 2 in let y = let z = 3 in z + x in y }";
        let module = parse_module(src).expect("parse module");
        assert!(type_check_module(&module).is_ok());
    }

    #[test]
    fn complex_calls_and_returns() {
        let src = "fn add1(a: i32) -> i32 { a + 1 } fn twice(a: i32) -> i32 { add1(add1(a)) } fn cond(b: bool) -> i32 { if b then 1 else 0 }";
        let module = parse_module(src).expect("parse module");
        assert!(type_check_module(&module).is_ok());
    }

    #[test]
    fn error_function_return_mismatch() {
        let src = "fn bad(a: i32) -> bool { a + 1 }";
        let module = parse_module(src).expect("parse module");
        let err = type_check_module(&module).unwrap_err();
        assert!(err.iter().any(|e| e.msg.contains("declared to return")));
    }
}
