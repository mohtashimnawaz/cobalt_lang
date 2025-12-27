use ariadne::{Report, ReportKind, Label, Source};
use crate::parser::ParseError;

/// Print parse errors using `ariadne` with spans and messages.
pub fn report_parse_errors(source: &str, filename: &str, errors: &[ParseError]) {
    for err in errors {
        let start = err.span.map(|s| s.start).unwrap_or(0);
        let end = err.span.map(|s| s.end).unwrap_or(start);

        let report = Report::build(ReportKind::Error, filename, start)
            .with_message(err.msg.clone())
            .with_label(Label::new((filename, start..end.max(start+1))).with_message(err.msg.clone()))
            .finish();

        // Print the report using a temporary Source for this filename
        let _ = report.print((filename, Source::from(source)));
    }
}
