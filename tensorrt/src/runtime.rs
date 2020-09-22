use tensorrt_sys::{create_infer_runtime, create_logger, delete_logger, destroy_infer_runtime, set_logger_severity};

#[repr(C)]
pub enum LoggerSeverity {
    InternalError,
    Error,
    Warning,
    Info,
    Verbose,
}

pub struct Logger {
    pub(crate) internal_logger: *mut tensorrt_sys::Logger_t,
}

impl Logger {
    pub fn new() -> Logger {
        let logger = unsafe { create_logger(LoggerSeverity::Warning as i32) };
        Logger {
            internal_logger: logger,
        }
    }

    pub fn severity(self, severity: LoggerSeverity) -> Logger {
        unsafe {
            set_logger_severity(self.internal_logger, severity as i32);
        };
        self
    }
}

impl Drop for Logger {
    fn drop(&mut self) {
        unsafe { delete_logger(self.internal_logger) };
    }
}

#[derive(Clone)]
pub struct Runtime {
    pub(crate) internal_runtime: *mut tensorrt_sys::Runtime_t,
}

impl Runtime {
    pub fn new(logger: &Logger) -> Runtime {
        let runtime = unsafe { create_infer_runtime(logger.internal_logger) };
        Runtime {
            internal_runtime: runtime,
        }
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        unsafe { destroy_infer_runtime(self.internal_runtime) };
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
