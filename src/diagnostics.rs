use ariadne::{Report, ReportKind, Label, Source, sources::SimpleCache};
use crate::parser::ParseError;

/// Print parse errors using `ariadne` with spans and messages.
pub fn report_parse_errors(source: &str, filename: &str, errors: &[ParseError]) {
    // Build a simple cache containing our file so `Report::print` can lookup the source
    let mut cache = SimpleCache::new();
    cache.add(filename, Source::from(source));

    for err in errors {
        let start = err.span.map(|s| s.start).unwrap_or(0);
        let end = err.span.map(|s| s.end).unwrap_or(start);

        let mut r = Report::build(ReportKind::Error, filename, start)
            .with_message(err.msg.clone());

        if start != end {
            r = r.with_label(Label::new(start..end).with_message(err.msg.clone()));
        } else {
            r = r.with_label(Label::new(start..start+1).with_message(err.msg.clone()));
        }

        let report = r.finish();
        let _ = report.print(&mut cache);
    }
}
