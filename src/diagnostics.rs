use ariadne::{Report, ReportKind, Label, Source};
use crate::parser::ParseError;
use crate::ty::{TypeError, TypeWarning};

/// Print parse errors using `ariadne` with spans and messages.
pub fn report_parse_errors(source: &str, filename: &str, errors: &[ParseError]) {
    for err in errors {
        let start = err.span.map(|s| s.start).unwrap_or(0);
        let end = err.span.map(|s| s.end).unwrap_or(start);

        let report: ariadne::Report<(String, std::ops::Range<usize>)> = Report::build(ReportKind::Error, filename.to_string(), start)
            .with_message(err.msg.clone())
            .with_label(Label::new((filename.to_string(), start..end.max(start+1))).with_message(err.msg.clone()))
            .finish();

        // Print the report using a temporary Source for this filename
        let _ = report.print((filename.to_string(), Source::from(source)));
    }
}

/// Print semantic/type errors using `ariadne` with spans and messages.
pub fn report_type_errors(source: &str, filename: &str, errors: &[TypeError]) {
    for err in errors {
        if let Some(span) = err.span {
            let start = span.start;
            let end = span.end;

            let report: ariadne::Report<(String, std::ops::Range<usize>)> = Report::build(ReportKind::Error, filename.to_string(), start)
                .with_message(err.msg.clone())
                .with_label(Label::new((filename.to_string(), start..end.max(start+1))).with_message(err.msg.clone()))
                .finish();

            let _ = report.print((filename.to_string(), Source::from(source)));
        } else {
            let start: usize = 0;
            let report: ariadne::Report<(String, std::ops::Range<usize>)> = Report::build(ReportKind::Error, filename.to_string(), start)
                .with_message(err.msg.clone())
                .finish();

            let _ = report.print((filename.to_string(), Source::from(source)));
        }
    }
}

/// Print semantic/type warnings using `ariadne` as warnings.
pub fn report_type_warnings(source: &str, filename: &str, warnings: &[TypeWarning]) {
    for w in warnings {
        if let Some(span) = w.span {
            let start = span.start;
            let end = span.end;

            let report: ariadne::Report<(String, std::ops::Range<usize>)> = Report::build(ReportKind::Warning, filename.to_string(), start)
                .with_message(w.msg.clone())
                .with_label(Label::new((filename.to_string(), start..end.max(start+1))).with_message(w.msg.clone()))
                .finish();

            let _ = report.print((filename.to_string(), Source::from(source)));
        } else {
            let start: usize = 0;
            let report: ariadne::Report<(String, std::ops::Range<usize>)> = Report::build(ReportKind::Warning, filename.to_string(), start)
                .with_message(w.msg.clone())
                .finish();

            let _ = report.print((filename.to_string(), Source::from(source)));
        }
    }
}
