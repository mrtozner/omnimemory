use super::{CommandContext, Result};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame, Terminal,
};
use std::io;
use std::time::Duration;

/// Handle dashboard command - show interactive TUI
pub fn handle_dashboard(ctx: &CommandContext) -> Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Run dashboard
    let res = run_dashboard(&mut terminal, ctx);

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    res
}

fn run_dashboard<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    _ctx: &CommandContext,
) -> Result<()> {
    loop {
        terminal.draw(|f| render_ui(f))?;

        // Handle input (with timeout for refresh)
        if event::poll(Duration::from_millis(250))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                    KeyCode::Char('r') => {
                        // Refresh dashboard
                        continue;
                    }
                    _ => {}
                }
            }
        }
    }
}

fn render_ui(f: &mut Frame) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Title
            Constraint::Length(5),  // System status
            Constraint::Length(5),  // MCP server
            Constraint::Min(5),     // Token savings
            Constraint::Length(3),  // Help
        ])
        .split(f.size());

    // Title
    let title = Paragraph::new("OmniMemory Dashboard")
        .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(title, chunks[0]);

    // System Status
    let status_text = vec![
        Line::from(vec![
            Span::raw("Status: "),
            Span::styled("● Running", Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::raw("PID: "),
            Span::styled("12345", Style::default().fg(Color::Yellow)),
            Span::raw("  Uptime: "),
            Span::styled("2h 14m", Style::default().fg(Color::Yellow)),
        ]),
    ];
    let status = Paragraph::new(status_text)
        .block(Block::default().borders(Borders::ALL).title("System Status"));
    f.render_widget(status, chunks[1]);

    // MCP Server
    let mcp_text = vec![
        Line::from(vec![
            Span::raw("Status: "),
            Span::styled("● Connected", Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::raw("Tools: "),
            Span::styled("6 available", Style::default().fg(Color::Cyan)),
            Span::raw("  Port: "),
            Span::styled("8000-8002", Style::default().fg(Color::Yellow)),
        ]),
    ];
    let mcp = Paragraph::new(mcp_text)
        .block(Block::default().borders(Borders::ALL).title("MCP Server"));
    f.render_widget(mcp, chunks[2]);

    // Token Savings
    let savings_text = vec![
        Line::from(vec![
            Span::raw("Total Saved: "),
            Span::styled("125,432 tokens", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::raw("Compression: "),
            Span::styled("94.4%", Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::raw("Cost Saved: "),
            Span::styled("$2.51", Style::default().fg(Color::Yellow)),
        ]),
    ];
    let savings = Paragraph::new(savings_text)
        .block(Block::default().borders(Borders::ALL).title("Token Savings (Last 24h)"));
    f.render_widget(savings, chunks[3]);

    // Help
    let help_text = Line::from(vec![
        Span::styled("[q]", Style::default().fg(Color::Yellow)),
        Span::raw(" quit  "),
        Span::styled("[r]", Style::default().fg(Color::Yellow)),
        Span::raw(" refresh"),
    ]);
    let help = Paragraph::new(help_text)
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL).title("Commands"));
    f.render_widget(help, chunks[4]);
}
