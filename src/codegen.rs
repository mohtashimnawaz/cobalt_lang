use anyhow::{anyhow, Result};

use crate::ast::Module;

/// Public API: compile a `Module` to LLVM IR as a String.
/// If the crate is built without the `llvm` feature this returns an error instructing how to enable it.
#[cfg(not(feature = "llvm"))]
pub fn compile_module_to_ir(module: &Module, _name: &str) -> Result<String> {
    // Fallback: emit simple C code for the module so local testing/demo can run without inkwell.
    // This is intentionally minimal and supports literals, top-level `let` bindings, simple binary add/sub/mul/div,
    // and explicit casts used in examples (float->int truncation, int->float promotion).
    use crate::ast::{Expr, Item, Literal, Type, BinaryExpr, BinaryOp};

    let mut out = String::new();
    out.push_str("#include <stdint.h>\n#include <stdbool.h>\n\n");

    // Emit globals for top-level lets (only support integer and float/bool literals for now)
    for item in &module.items {
        if let Item::Let { name, ty, value, .. } = item {
            match value {
                Expr::Literal(Literal::Int(i)) => {
                    match ty {
                        Some(Type::I64) => out.push_str(&format!("long long {} = {}LL;\n", name, i)),
                        Some(Type::I32) | None => out.push_str(&format!("int {} = {};\n", name, i)),
                        _ => out.push_str(&format!("int {} = {};\n", name, i)),
                    }
                }
                Expr::Literal(Literal::Float(f)) => {
                    match ty {
                        Some(Type::F32) => out.push_str(&format!("float {} = {}f;\n", name, f)),
                        _ => out.push_str(&format!("double {} = {};\n", name, f)),
                    }
                }
                Expr::Literal(Literal::Bool(b)) => {
                    out.push_str(&format!("int {} = {};\n", name, if *b {1} else {0}));
                }
                _ => {
                    // unsupported initializer; default to 0
                    out.push_str(&format!("int {} = 0;\n", name));
                }
            }
        }
    }
    out.push_str("\n");

    // Helper to emit expressions as C and return the C type as string
    fn emit_expr(e: &Expr, out: &mut String) -> String {
        match e {
            Expr::Literal(Literal::Int(i)) => { out.push_str(&format!("{}", i)); "int".into() }
            Expr::Literal(Literal::Float(f)) => { out.push_str(&format!("{}", f)); "double".into() }
            Expr::Literal(Literal::Bool(b)) => { out.push_str(&format!("{}", if *b {1} else {0})); "int".into() }
            Expr::Var(name) => { out.push_str(name); "int".into() }
            Expr::Binary(BinaryExpr { op, left, right, .. }) => {
                out.push_str("(");
                let _ = emit_expr(left, out);
                match op {
                    BinaryOp::Add => out.push_str(" + "),
                    BinaryOp::Sub => out.push_str(" - "),
                    BinaryOp::Mul => out.push_str(" * "),
                    BinaryOp::Div => out.push_str(" / "),
                    _ => out.push_str(" /*op*/ "),
                }
                let _ = emit_expr(right, out);
                out.push_str(")");
                "double".into() // choose double for numeric ops to be safe
            }
            Expr::Cast { expr, ty, .. } => {
                out.push_str("(");
                match ty {
                    Type::I32 => out.push_str("(int)"),
                    Type::I64 => out.push_str("(long long)"),
                    Type::F32 => out.push_str("(float)"),
                    Type::F64 => out.push_str("(double)"),
                    Type::Bool => out.push_str("(int)"),
                    _ => out.push_str("/*cast*/"),
                }
                let res = emit_expr(expr, out);
                out.push_str(")");
                match ty {
                    Type::I32 => "int".into(),
                    Type::I64 => "long long".into(),
                    Type::F32 => "float".into(),
                    Type::F64 => "double".into(),
                    Type::Bool => "int".into(),
                    _ => res,
                }
            }
            Expr::Call { callee, args, .. } => {
                if let Expr::Var(fn_name) = &**callee {
                    out.push_str(&format!("{}(", fn_name));
                    for (i, a) in args.iter().enumerate() {
                        if i>0 { out.push_str(", "); }
                        let _ = emit_expr(a, out);
                    }
                    out.push_str(")");
                } else {
                    out.push_str("/*call expr*/");
                }
                "int".into()
            }
            Expr::If { cond, then_branch, else_branch, .. } => {
                out.push_str("(");
                let _ = emit_expr(cond, out);
                out.push_str(" ? ");
                let _ = emit_expr(then_branch, out);
                out.push_str(" : ");
                let _ = emit_expr(else_branch, out);
                out.push_str(")");
                "int".into()
            }
            Expr::Let { .. } => { out.push_str("/*let*/ 0"); "int".into() }
        }
    }

    // Emit functions
    for item in &module.items {
        if let Item::Function { name, params, ret_type, body, .. } = item {
            // Map types
            let c_ret = match ret_type {
                Type::I32 => "int",
                Type::I64 => "long long",
                Type::F32 => "float",
                Type::F64 => "double",
                Type::Bool => "int",
                _ => "int",
            };
            out.push_str(&format!("{} {}(", c_ret, name));
            for (i, p) in params.iter().enumerate() {
                if i>0 { out.push_str(", "); }
                let c_ty = match p.ty {
                    Type::I32 => "int",
                    Type::I64 => "long long",
                    Type::F32 => "float",
                    Type::F64 => "double",
                    Type::Bool => "int",
                    _ => "int",
                };
                out.push_str(&format!("{} {}", c_ty, p.name));
            }
            out.push_str(") {\n    return ");
            let _ = emit_expr(body, &mut out);
            out.push_str(";\n}\n\n");
        }
    }

    Ok(out)
}

#[cfg(feature = "llvm")]
mod llvm_codegen {
    use super::*;
    use std::collections::HashMap;
    use crate::ast::{BinaryExpr, BinaryOp, Expr, Item, Literal, Param, Type};
    use inkwell::builder::Builder;
    use inkwell::context::Context;
    use inkwell::module::Module as LLVMModule;
    use inkwell::types::{BasicTypeEnum, FunctionType};
    use inkwell::values::{BasicValueEnum, FunctionValue, PointerValue};
    use inkwell::AddressSpace;

    pub struct Codegen<'ctx> {
        pub context: &'ctx Context,
        pub module: LLVMModule<'ctx>,
        pub builder: Builder<'ctx>,
        /// Function prototypes for top-level functions (so we can call before a function is defined)
        pub fn_protos: HashMap<String, FunctionValue<'ctx>>,
        /// Variable scopes (maps names to allocas)
        pub vars: Vec<HashMap<String, PointerValue<'ctx>>>,
    }

    impl<'ctx> Codegen<'ctx> {
        pub fn new(context: &'ctx Context, name: &str) -> Self {
            let module = context.create_module(name);
            let builder = context.create_builder();
            Self { context, module, builder, fn_protos: HashMap::new(), vars: Vec::new() }
        }

        fn i32_type(&self) -> inkwell::types::IntType<'ctx> {
            self.context.i32_type()
        }

        fn bool_type(&self) -> inkwell::types::IntType<'ctx> {
            self.context.bool_type()
        }

        fn to_llvm_type(&self, ty: &Type) -> BasicTypeEnum<'ctx> {
            match ty {
                Type::I32 => self.i32_type().into(),
                Type::I64 => self.context.i64_type().into(),
                Type::F32 => self.context.f32_type().into(),
                Type::F64 => self.context.f64_type().into(),
                Type::Bool => self.bool_type().into(),
                Type::Func { params, ret } => {
                    let param_types: Vec<BasicTypeEnum> = params.iter().map(|p| self.to_llvm_type(p)).collect();
                    let ret_type = self.to_llvm_type(ret);
                    ret_type.fn_type(&param_types.iter().map(|t| *t).collect::<Vec<_>>(), false).as_basic_type_enum()
                }
            }
        }

        /// Create function prototypes for all top-level functions first
        fn declare_functions(&mut self, module: &Module) {
            for item in &module.items {
                if let Item::Function { name, params, ret_type, .. } = item {
                    let param_types: Vec<BasicTypeEnum> = params.iter().map(|p| self.to_llvm_type(&p.ty)).collect();
                    let ret_type = self.to_llvm_type(ret_type);
                    let fn_type = match ret_type {
                        BasicTypeEnum::IntType(int_ty) => int_ty.fn_type(&param_types, false),
                        BasicTypeEnum::PointerType(ptr_ty) => ptr_ty.fn_type(&param_types, false),
                        BasicTypeEnum::FloatType(f) => f.fn_type(&param_types, false),
                        BasicTypeEnum::StructType(s) => s.fn_type(&param_types, false),
                        BasicTypeEnum::VectorType(v) => v.fn_type(&param_types, false),
                    };

                    let function = self.module.add_function(name, fn_type, None);
                    self.fn_protos.insert(name.clone(), function);
                }
            }
        }

        pub fn compile_module(&mut self, module: &Module) -> Result<()> {
            // Declare functions so recursive calls work
            self.declare_functions(module);

            for item in &module.items {
                match item {
                    Item::Function { name, params, ret_type, body, .. } => {
                        self.compile_function(name, params, ret_type, body)
                            .with_context(|| format!("failed to compile function `{}`", name))?;
                    }
                    Item::Let { name, value, .. } => {
                        // Top-level immutable binding -> emit as global constant (only i32/bool supported for now)
                        self.compile_top_level_let(name, value)?;
                    }
                }
            }

            Ok(())
        }

        fn compile_top_level_let(&mut self, name: &str, value: &Expr) -> Result<()> {
            let evaluated = self.compile_expr(value)?;
            match evaluated {
                BasicValueEnum::IntValue(iv) => {
                    let g = self.module.add_global(iv.get_type(), None, name);
                    g.set_initializer(&iv);
                }
                BasicValueEnum::IntVectorValue(_) | BasicValueEnum::FloatValue(_) | BasicValueEnum::PointerValue(_) => {
                    // For MVP, only support integer globals
                    return Err(anyhow!("only integer globals are supported for top-level `let` at the moment"));
                }
            }
            Ok(())
        }

        fn compile_function(&mut self, name: &str, params: &Vec<Param>, ret_ty: &Type, body: &Expr) -> Result<()> {
            let function = *self.fn_protos.get(name).ok_or_else(|| anyhow!("function prototype not found"))?;

            // Create a basic block and position the builder there
            let entry = self.context.append_basic_block(function, "entry");
            self.builder.position_at_end(entry);

            // Create a new variable scope for this function
            self.vars.push(HashMap::new());

            // Allocate space for parameters and store incoming values
            for (i, param) in params.iter().enumerate() {
                let llvm_ty = self.to_llvm_type(&param.ty);
                let alloca = self.create_entry_block_alloca(function, &param.name, llvm_ty);
                let incoming = function.get_nth_param(i as u32).unwrap();
                incoming.set_name(&param.name);
                self.builder.build_store(alloca, incoming);
                self.vars.last_mut().unwrap().insert(param.name.clone(), alloca);
            }

            // Compile function body
            let ret_val = self.compile_expr(body)?;

            // Ensure return instruction
            match ret_val {
                BasicValueEnum::IntValue(iv) => { self.builder.build_return(Some(&iv)); }
                BasicValueEnum::FloatValue(fv) => { self.builder.build_return(Some(&fv)); }
                BasicValueEnum::PointerValue(pv) => { self.builder.build_return(Some(&pv)); }
                BasicValueEnum::IntVectorValue(_) => { return Err(anyhow!("unsupported return vector type")); }
            }

            // Pop scope
            self.vars.pop();

            Ok(())
        }

        fn get_var(&self, name: &str) -> Option<PointerValue<'ctx>> {
            for scope in self.vars.iter().rev() {
                if let Some(v) = scope.get(name) {
                    return Some(*v);
                }
            }
            None
        }

        fn compile_expr(&mut self, expr: &Expr) -> Result<BasicValueEnum<'ctx>> {
            match expr {
                    Expr::Literal(Literal::Int(i)) => Ok(self.i32_type().const_int(*i as u64, true).into()),
                Expr::Literal(Literal::Bool(b)) => Ok(self.bool_type().const_int(if *b {1} else {0}, false).into()),
                Expr::Literal(Literal::Float(f)) => Ok(self.context.f64_type().const_float(*f).into()),
                Expr::Var(name) => {
                    if let Some(ptr) = self.get_var(name) {
                        Ok(self.builder.build_load(ptr, &format!("load_{}", name)))
                    } else if let Some(g) = self.module.get_global(name) {
                        Ok(g.get_initializer().unwrap())
                    } else if let Some(fnval) = self.fn_protos.get(name) {
                        // A function value as pointer
                        Ok(fnval.as_global_value().as_pointer_value().into())
                    } else {
                        Err(anyhow!("unknown variable `{}`", name))
                    }
                }
                Expr::Binary(BinaryExpr { op, left, right, .. }) => {
                    let l = self.compile_expr(left)?;
                    let r = self.compile_expr(right)?;
                    // Handle float-int mixed cases: convert ints to floats when either operand is float.
                    if l.is_float_value() || r.is_float_value() {
                        // Determine target float type (f64 if either is f64)
                        let target_is_f64 = (l.is_float_value() && l.into_float_value().get_type().get_bit_width() == 64) || (r.is_float_value() && r.into_float_value().get_type().get_bit_width() == 64);
                        let (lf, rf) = {
                            // convert left to float if needed
                            let lf = if l.is_float_value() {
                                let fv = l.into_float_value();
                                if target_is_f64 && fv.get_type().get_bit_width() == 32 {
                                    self.builder.build_float_ext(fv, self.context.f64_type(), "fexttmp")
                                } else if !target_is_f64 && fv.get_type().get_bit_width() == 64 {
                                    self.builder.build_float_trunc(fv, self.context.f32_type(), "ftrunctmp")
                                } else {
                                    fv
                                }
                            } else {
                                // int -> float
                                let iv = l.into_int_value();
                                if target_is_f64 {
                                    self.builder.build_signed_int_to_float(iv, self.context.f64_type(), "sitofptmp")
                                } else {
                                    self.builder.build_signed_int_to_float(iv, self.context.f32_type(), "sitofptmp")
                                }
                            };
                            // convert right to float if needed
                            let rf = if r.is_float_value() {
                                let fv = r.into_float_value();
                                if target_is_f64 && fv.get_type().get_bit_width() == 32 {
                                    self.builder.build_float_ext(fv, self.context.f64_type(), "fexttmp")
                                } else if !target_is_f64 && fv.get_type().get_bit_width() == 64 {
                                    self.builder.build_float_trunc(fv, self.context.f32_type(), "ftrunctmp")
                                } else {
                                    fv
                                }
                            } else {
                                let iv = r.into_int_value();
                                if target_is_f64 {
                                    self.builder.build_signed_int_to_float(iv, self.context.f64_type(), "sitofptmp")
                                } else {
                                    self.builder.build_signed_int_to_float(iv, self.context.f32_type(), "sitofptmp")
                                }
                            };
                            (lf, rf)
                        };
                        match op {
                            BinaryOp::Add => Ok(self.builder.build_float_add(lf, rf, "faddtmp").into()),
                            BinaryOp::Sub => Ok(self.builder.build_float_sub(lf, rf, "fsubtmp").into()),
                            BinaryOp::Mul => Ok(self.builder.build_float_mul(lf, rf, "fmultmp").into()),
                            BinaryOp::Div => Ok(self.builder.build_float_div(lf, rf, "fdivtmp").into()),
                            BinaryOp::Eq => Ok(self.builder.build_float_compare(inkwell::FloatPredicate::OEQ, lf, rf, "feqtmp").into()),
                            BinaryOp::Ne => Ok(self.builder.build_float_compare(inkwell::FloatPredicate::ONE, lf, rf, "fnetmp").into()),
                            BinaryOp::Lt => Ok(self.builder.build_float_compare(inkwell::FloatPredicate::OLT, lf, rf, "flttmp").into()),
                            BinaryOp::Le => Ok(self.builder.build_float_compare(inkwell::FloatPredicate::OLE, lf, rf, "fletmp").into()),
                            BinaryOp::Gt => Ok(self.builder.build_float_compare(inkwell::FloatPredicate::OGT, lf, rf, "fgttmp").into()),
                            BinaryOp::Ge => Ok(self.builder.build_float_compare(inkwell::FloatPredicate::OGE, lf, rf, "fgetmp").into()),
                            BinaryOp::And | BinaryOp::Or => Err(anyhow!("logical operators not valid for floats")),
                        }
                    } else {
                        let li = l.into_int_value();
                        let ri = r.into_int_value();
                        match op {
                            BinaryOp::Add => Ok(self.builder.build_int_add(li, ri, "addtmp").into()),
                            BinaryOp::Sub => Ok(self.builder.build_int_sub(li, ri, "subtmp").into()),
                            BinaryOp::Mul => Ok(self.builder.build_int_mul(li, ri, "multmp").into()),
                            BinaryOp::Div => Ok(self.builder.build_int_signed_div(li, ri, "divtmp").into()),
                            BinaryOp::Eq => Ok(self.builder.build_int_compare(inkwell::IntPredicate::EQ, li, ri, "eqtmp").into()),
                            BinaryOp::Ne => Ok(self.builder.build_int_compare(inkwell::IntPredicate::NE, li, ri, "netmp").into()),
                            BinaryOp::Lt => Ok(self.builder.build_int_compare(inkwell::IntPredicate::SLT, li, ri, "lttmp").into()),
                            BinaryOp::Le => Ok(self.builder.build_int_compare(inkwell::IntPredicate::SLE, li, ri, "letmp").into()),
                            BinaryOp::Gt => Ok(self.builder.build_int_compare(inkwell::IntPredicate::SGT, li, ri, "gttmp").into()),
                            BinaryOp::Ge => Ok(self.builder.build_int_compare(inkwell::IntPredicate::SGE, li, ri, "getmp").into()),
                            BinaryOp::And => Ok(self.builder.build_and(li, ri, "andtmp").into()),
                            BinaryOp::Or => Ok(self.builder.build_or(li, ri, "ortmp").into()),
                        }
                    }
                }
                Expr::If { cond, then_branch, else_branch, .. } => {
                    let cond_v = self.compile_expr(cond)?;
                    let cond_bool = cond_v.into_int_value();

                    let function = self.builder.get_insert_block().unwrap().get_parent().unwrap();

                    let then_bb = self.context.append_basic_block(function, "then");
                    let else_bb = self.context.append_basic_block(function, "else");
                    let cont_bb = self.context.append_basic_block(function, "ifcont");

                    self.builder.build_conditional_branch(cond_bool, then_bb, else_bb);

                    // Then block
                    self.builder.position_at_end(then_bb);
                    let then_val = self.compile_expr(then_branch)?;
                    self.builder.build_unconditional_branch(cont_bb);
                    let then_end = self.builder.get_insert_block().unwrap();

                    // Else block
                    self.builder.position_at_end(else_bb);
                    let else_val = self.compile_expr(else_branch)?;
                    self.builder.build_unconditional_branch(cont_bb);
                    let else_end = self.builder.get_insert_block().unwrap();

                    // Continue block
                    self.builder.position_at_end(cont_bb);

                    // Create phi to merge then/else values
                    let phi = self.builder.build_phi(then_val.get_type(), "iftmp");
                    phi.add_incoming(&[(&then_val, then_end), (&else_val, else_end)]);
                    Ok(phi.as_basic_value())
                }
                Expr::Let { name, value, body, .. } => {
                    // Evaluate value
                    let val = self.compile_expr(value)?;
                    // Allocate in entry block
                    let function = self.builder.get_insert_block().unwrap().get_parent().unwrap();
                    let alloca = self.create_entry_block_alloca(function, name, val.get_type());
                    self.builder.build_store(alloca, val);
                    // Create new scope
                    self.vars.push(HashMap::new());
                    self.vars.last_mut().unwrap().insert(name.clone(), alloca);
                    let result = self.compile_expr(body)?;
                    self.vars.pop();
                    Ok(result)
                }
                Expr::Call { callee, args, .. } => {
                    // We only support callee being a variable referring to a function
                    if let Expr::Var(fn_name) = &**callee {
                        let fn_val = *self.fn_protos.get(fn_name).ok_or_else(|| anyhow!("unknown function `{}`", fn_name))?;
                        let mut compiled_args: Vec<BasicValueEnum> = Vec::new();
                        for a in args {
                            compiled_args.push(self.compile_expr(a)?);
                        }
                        let call_site = self.builder.build_call(fn_val, &compiled_args.iter().map(|v| *v).collect::<Vec<_>>(), "calltmp");
                        match call_site.try_as_basic_value().left() {
                            Some(v) => Ok(v),
                            None => Err(anyhow!("function did not return a value or returned void")),
                        }
                    } else {
                        Err(anyhow!("call to non-function expression not supported in MVP"))
                    }
                }
                Expr::Cast { expr, ty, .. } => self.compile_cast(expr, ty.clone()),
                Expr::Binary(_) => unreachable!("handled above"),
                Expr::Call { .. } => unreachable!("handled above"),
            }
        }

        fn create_entry_block_alloca(&self, function: FunctionValue<'ctx>, name: &str, ty: BasicTypeEnum<'ctx>) -> PointerValue<'ctx> {
            let entry = function.get_first_basic_block().unwrap();
            let builder = self.context.create_builder();
            match entry.get_first_instruction() {
                Some(instr) => builder.position_before(&instr),
                None => builder.position_at_end(entry),
            }
            builder.build_alloca(ty, name)
        }

        fn compile_cast(&mut self, expr: &Expr, ty: Type) -> Result<BasicValueEnum<'ctx>> {
            let val = self.compile_expr(expr)?;
            match val {
                BasicValueEnum::IntValue(iv) => {
                    let src_bits = iv.get_type().get_bit_width();
                    match ty {
                        Type::I32 => {
                            if src_bits == 32 { Ok(iv.into()) }
                            else if src_bits < 32 { Ok(self.builder.build_int_s_extend(iv, self.i32_type(), "sexttmp").into()) }
                            else { Ok(self.builder.build_int_truncate(iv, self.i32_type(), "trunctmp").into()) }
                        }
                        Type::I64 => {
                            if src_bits == 64 { Ok(iv.into()) }
                            else if src_bits < 64 { Ok(self.builder.build_int_s_extend(iv, self.context.i64_type(), "sexttmp").into()) }
                            else { Ok(self.builder.build_int_truncate(iv, self.context.i64_type(), "trunctmp").into()) }
                        }
                        Type::F32 => Ok(self.builder.build_signed_int_to_float(iv, self.context.f32_type(), "sitofptmp").into()),
                        Type::F64 => Ok(self.builder.build_signed_int_to_float(iv, self.context.f64_type(), "sitofptmp").into()),
                        Type::Bool => {
                            let zero = iv.get_type().const_int(0, false);
                            Ok(self.builder.build_int_compare(inkwell::IntPredicate::NE, iv, zero, "boolcast").into())
                        }
                        Type::Func { .. } => Err(anyhow!("cannot cast integer to function type")),
                    }
                }
                BasicValueEnum::FloatValue(fv) => {
                    let src_bits = fv.get_type().get_bit_width();
                    match ty {
                        Type::F64 => {
                            if src_bits == 64 { Ok(fv.into()) }
                            else { Ok(self.builder.build_float_ext(fv, self.context.f64_type(), "fexttmp").into()) }
                        }
                        Type::F32 => {
                            if src_bits == 32 { Ok(fv.into()) }
                            else { Ok(self.builder.build_float_trunc(fv, self.context.f32_type(), "ftrunctmp").into()) }
                        }
                        Type::I32 => Ok(self.builder.build_float_to_signed_int(fv, self.i32_type(), "fptositmp").into()),
                        Type::I64 => Ok(self.builder.build_float_to_signed_int(fv, self.context.i64_type(), "fptositmp").into()),
                        Type::Bool => {
                            let zero = fv.get_type().const_float(0.0);
                            Ok(self.builder.build_float_compare(inkwell::FloatPredicate::ONE, fv, zero, "boolcast").into())
                        }
                        Type::Func { .. } => Err(anyhow!("cannot cast float to function type")),
                    }
                }
                BasicValueEnum::PointerValue(_) => Err(anyhow!("cannot cast pointer values in MVP")),
                BasicValueEnum::IntVectorValue(_) => Err(anyhow!("vector casts not supported")),
            }
        }
    }

    /// Public wrapper that compiles and returns textual LLVM IR
    pub fn compile_module_to_ir(module: &Module, name: &str) -> Result<String> {
        let context = Context::create();
        let mut cg = Codegen::new(&context, name);
        cg.compile_module(module)?;
        Ok(cg.module.print_to_string().to_string())
    }

    /// Compile to an object file using the host target
    pub fn compile_module_to_object(module: &Module, name: &str, path: &std::path::Path) -> Result<()> {
        use inkwell::targets::{Target, InitializationConfig, FileType, RelocMode, CodeModel};
        let context = Context::create();
        let mut cg = Codegen::new(&context, name);
        cg.compile_module(module)?;

        // Initialize targets
        Target::initialize_all(&InitializationConfig::default());
        let triple = inkwell::targets::TargetMachine::get_default_triple();
        let target = Target::from_triple(&triple);
        let cpu = "generic";
        let features = "";
        let tm = target.create_target_machine(&triple, cpu, features, inkwell::OptimizationLevel::Default, RelocMode::Default, CodeModel::Default)
            .ok_or_else(|| anyhow!("failed to create target machine"))?;

        tm.write_to_file(&cg.module, FileType::Object, path)
            .with_context(|| format!("failed to write object file to {:?}", path))?;
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::ast::*;

        #[test]
        fn compile_simple_add() {
            let m = Module { items: vec![Item::Function { name: "add".to_string(), params: vec![Param { name: "a".to_string(), ty: Type::I32 }, Param { name: "b".to_string(), ty: Type::I32 }], ret_type: Type::I32, body: Expr::Binary(BinaryExpr { op: BinaryOp::Add, left: Box::new(Expr::Var("a".to_string())), right: Box::new(Expr::Var("b".to_string())), span: None }), span: None }] };
            let ir = compile_module_to_ir(&m, "test").expect("codegen");
            assert!(ir.contains("define i32 @add(i32 %"));
            assert!(ir.contains("add i32"));
        }

        #[test]
        fn compile_cast_int_to_float() {
            let m = Module { items: vec![Item::Function { name: "to_double".to_string(), params: vec![Param { name: "x".to_string(), ty: Type::I32 }], ret_type: Type::F64, body: Expr::Cast { expr: Box::new(Expr::Var("x".to_string())), ty: Type::F64, span: None }, span: None }] };
            let ir = compile_module_to_ir(&m, "test_cast").expect("codegen");
            assert!(ir.contains("define double @to_double(i32 %"));
            // Ensure an int-to-float conversion instruction (sitofp) appears
            assert!(ir.contains("sitofp i32"));
        }

        #[test]
        fn compile_add_int_and_float() {
            let m = Module { items: vec![Item::Function { name: "to_double".to_string(), params: vec![Param { name: "x".to_string(), ty: Type::I32 }], ret_type: Type::F64, body: Expr::Binary(BinaryExpr { op: BinaryOp::Add, left: Box::new(Expr::Var("x".to_string())), right: Box::new(Expr::Literal(Literal::Float(1.5))), span: None }), span: None }] };
            let ir = compile_module_to_ir(&m, "test_add").expect("codegen");
            // int -> float conversion should be present and then an fadd
            assert!(ir.contains("sitofp i32"));
            assert!(ir.contains("fadd"));
        }

        #[test]
        fn compile_cast_float_to_int() {
            let m = Module { items: vec![Item::Function { name: "to_int".to_string(), params: vec![], ret_type: Type::I32, body: Expr::Cast { expr: Box::new(Expr::Literal(Literal::Float(2.5))), ty: Type::I32, span: None }, span: None }] };
            let ir = compile_module_to_ir(&m, "test_cast_f2i").expect("codegen");
            // Should contain a float-to-int conversion instruction (fptosi)
            assert!(ir.contains("fptosi"));
        }

        #[test]
        fn compile_cast_i64_to_f32() {
            let m = Module { items: vec![Item::Function { name: "i64_to_f32".to_string(), params: vec![Param { name: "x".to_string(), ty: Type::I64 }], ret_type: Type::F32, body: Expr::Cast { expr: Box::new(Expr::Var("x".to_string())), ty: Type::F32, span: None }, span: None }] };
            let ir = compile_module_to_ir(&m, "test_cast_i64_f32").expect("codegen");
            // int64 -> float32 conversion should contain sitofp i64
            assert!(ir.contains("sitofp i64"));
        }

        #[test]
        fn compile_add_i64_and_f64() {
            let m = Module { items: vec![Item::Function { name: "add_i64_f64".to_string(), params: vec![Param { name: "x".to_string(), ty: Type::I64 }], ret_type: Type::F64, body: Expr::Binary(BinaryExpr { op: BinaryOp::Add, left: Box::new(Expr::Var("x".to_string())), right: Box::new(Expr::Literal(Literal::Float(1.5))), span: None }), span: None }] };
            let ir = compile_module_to_ir(&m, "test_add_i64_f64").expect("codegen");
            // i64 -> f64 conversion should be present and then an fadd
            assert!(ir.contains("sitofp i64"));
            assert!(ir.contains("fadd"));
        }

        #[test]
        fn compile_cast_float_to_bool() {
            let m = Module { items: vec![Item::Function { name: "f2b".to_string(), params: vec![], ret_type: Type::Bool, body: Expr::Cast { expr: Box::new(Expr::Literal(Literal::Float(2.5))), ty: Type::Bool, span: None }, span: None }] };
            let ir = compile_module_to_ir(&m, "test_cast_f2b").expect("codegen");
            // Should contain a float compare (fcmp) for bool cast
            assert!(ir.contains("fcmp") || ir.contains("fcmp one"));
        }

        #[test]
        fn compile_cast_i32_to_bool() {
            let m = Module { items: vec![Item::Function { name: "i2b".to_string(), params: vec![Param { name: "x".to_string(), ty: Type::I32 }], ret_type: Type::Bool, body: Expr::Cast { expr: Box::new(Expr::Var("x".to_string())), ty: Type::Bool, span: None }, span: None }] };
            let ir = compile_module_to_ir(&m, "test_cast_i2b").expect("codegen");
            // Should contain an integer compare (icmp ne)
            assert!(ir.contains("icmp") || ir.contains("icmp ne"));
        }

        // Runtime integration test: compile to object, link with clang, run and assert runtime result
        #[test]
        fn runtime_integration_casts() {
            use std::fs::File;
            use std::io::Write;
            use std::process::Command;
            use std::env::temp_dir;
            // Function converts a float 2.5 -> i32 (should truncate to 2)
            let m = Module { items: vec![Item::Function { name: "to_int".to_string(), params: vec![], ret_type: Type::I32, body: Expr::Cast { expr: Box::new(Expr::Literal(Literal::Float(2.5))), ty: Type::I32, span: None }, span: None }] };
            let td = temp_dir().join(format!("cobalt_runtime_test_{}", std::process::id()));
            let _ = std::fs::create_dir_all(&td);
            let obj_path = td.join("module.o");
            compile_module_to_object(&m, "test_runtime", &obj_path).expect("compile to object");
            // write a small C main that calls to_int and returns its result
            let main_c = td.join("main.c");
            let mut f = File::create(&main_c).expect("create main.c");
            f.write_all(b"extern int to_int(); int main(){ return to_int(); }").expect("write main.c");
            // use clang to link
            let clang = std::env::var("CC").unwrap_or_else(|_| "clang".to_string());
            let out_exe = td.join("run_test");
            let status = Command::new(&clang)
                .args(&[obj_path.to_str().unwrap(), main_c.to_str().unwrap(), "-o", out_exe.to_str().unwrap()])
                .status()
                .expect("link with clang");
            assert!(status.success(), "link failed");
            let run_status = Command::new(out_exe.to_str().unwrap())
                .status()
                .expect("run test");
            // main returns 2 (truncated from 2.5)
            assert_eq!(run_status.code(), Some(2));
        }
    }
}

#[cfg(feature = "llvm")]
pub use llvm_codegen::compile_module_to_ir;

#[cfg(feature = "llvm")]
pub use llvm_codegen::compile_module_to_object;
