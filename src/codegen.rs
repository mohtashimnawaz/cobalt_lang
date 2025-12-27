use anyhow::{anyhow, Result};

use crate::ast::{Module, Type};

/// Public API: compile a `Module` to LLVM IR as a String.
/// If the crate is built without the `llvm` feature this returns an error instructing how to enable it.
#[cfg(not(feature = "llvm"))]
pub fn compile_module_to_ir(_module: &Module, _name: &str) -> Result<String> {
    Err(anyhow!("`llvm` feature is disabled. Build with `--features llvm` to enable codegen (requires LLVM/clang on your system)."))
}

#[cfg(feature = "llvm")]
mod llvm_codegen {
    use super::*;
    use std::collections::HashMap;
    use crate::ast::{BinaryExpr, BinaryOp, Expr, Item, Literal, Param};
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
                    match op {
                        BinaryOp::Add => Ok(self.builder.build_int_add(l.into_int_value(), r.into_int_value(), "addtmp").into()),
                        BinaryOp::Sub => Ok(self.builder.build_int_sub(l.into_int_value(), r.into_int_value(), "subtmp").into()),
                        BinaryOp::Mul => Ok(self.builder.build_int_mul(l.into_int_value(), r.into_int_value(), "multmp").into()),
                        BinaryOp::Div => Ok(self.builder.build_int_signed_div(l.into_int_value(), r.into_int_value(), "divtmp").into()),
                        BinaryOp::Eq => Ok(self.builder.build_int_compare(inkwell::IntPredicate::EQ, l.into_int_value(), r.into_int_value(), "eqtmp").into()),
                        BinaryOp::Ne => Ok(self.builder.build_int_compare(inkwell::IntPredicate::NE, l.into_int_value(), r.into_int_value(), "netmp").into()),
                        BinaryOp::Lt => Ok(self.builder.build_int_compare(inkwell::IntPredicate::SLT, l.into_int_value(), r.into_int_value(), "lttmp").into()),
                        BinaryOp::Le => Ok(self.builder.build_int_compare(inkwell::IntPredicate::SLE, l.into_int_value(), r.into_int_value(), "letmp").into()),
                        BinaryOp::Gt => Ok(self.builder.build_int_compare(inkwell::IntPredicate::SGT, l.into_int_value(), r.into_int_value(), "gttmp").into()),
                        BinaryOp::Ge => Ok(self.builder.build_int_compare(inkwell::IntPredicate::SGE, l.into_int_value(), r.into_int_value(), "getmp").into()),
                        BinaryOp::And => Ok(self.builder.build_and(l.into_int_value(), r.into_int_value(), "andtmp").into()),
                        BinaryOp::Or => Ok(self.builder.build_or(l.into_int_value(), r.into_int_value(), "ortmp").into()),
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
    }
}

#[cfg(feature = "llvm")]
pub use llvm_codegen::compile_module_to_ir;

#[cfg(feature = "llvm")]
pub use llvm_codegen::compile_module_to_object;
