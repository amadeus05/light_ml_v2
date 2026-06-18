LABEL_COVERAGE_HTML_TEMPLATE = """<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #070b16;
      --panel: #101827;
      --panel-2: #131e31;
      --text: #f5f7fb;
      --muted: #94a3b8;
      --line: rgba(148, 163, 184, 0.18);
      --green: #34d399;
      --red: #fb7185;
      --yellow: #fbbf24;
      --blue: #60a5fa;
      --purple: #a78bfa;
      --shadow: 0 24px 80px rgba(0, 0, 0, 0.36);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background:
        linear-gradient(180deg, rgba(96,165,250,0.09), transparent 36rem),
        radial-gradient(circle at top left, rgba(52, 211, 153, 0.13), transparent 34rem),
        radial-gradient(circle at top right, rgba(167, 139, 250, 0.16), transparent 30rem),
        var(--bg);
    }}
    main {{ width: min(1320px, calc(100vw - 44px)); margin: 0 auto; padding: 38px 0 60px; }}
    header {{ display: flex; justify-content: space-between; gap: 24px; align-items: flex-start; margin-bottom: 28px; }}
    h1 {{ margin: 0 0 10px; font-size: 42px; line-height: 1.05; letter-spacing: -0.055em; }}
    h2 {{ margin: 0 0 16px; font-size: 19px; letter-spacing: -0.025em; }}
    h3 {{ margin: 0 0 10px; font-size: 15px; color: var(--muted); font-weight: 600; }}
    p {{ margin: 0; color: var(--muted); line-height: 1.55; }}
    .badge {{
      display: inline-flex; align-items: center; gap: 8px;
      padding: 9px 12px; border: 1px solid var(--line); border-radius: 999px;
      background: rgba(15, 23, 42, 0.82); color: var(--muted); white-space: nowrap;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.06);
    }}
    .grid {{ display: grid; gap: 18px; }}
    .cards {{ grid-template-columns: repeat(4, minmax(0, 1fr)); margin-bottom: 20px; }}
    .two {{ grid-template-columns: 1fr 1fr; }}
    .panel {{
      position: relative; overflow: hidden;
      border: 1px solid var(--line); border-radius: 26px; padding: 20px;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.075), rgba(255,255,255,0.026)),
        rgba(15, 23, 42, 0.88);
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }}
    .panel::before {{
      content: ""; position: absolute; inset: 0 0 auto; height: 1px;
      background: linear-gradient(90deg, transparent, rgba(255,255,255,0.28), transparent);
    }}
    .card {{ min-height: 132px; }}
    .card .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 10px; }}
    .card .value {{ font-size: 32px; font-weight: 800; letter-spacing: -0.045em; }}
    .card .hint {{ margin-top: 8px; color: var(--muted); font-size: 12px; }}
    .green {{ color: var(--green); }} .red {{ color: var(--red); }} .yellow {{ color: var(--yellow); }}
    .blue {{ color: var(--blue); }} .purple {{ color: var(--purple); }}
    .section {{ margin-top: 20px; }}
    .chart {{
      width: 100%; overflow: hidden; border-radius: 20px;
      background:
        linear-gradient(180deg, rgba(15,23,42,0.96), rgba(6,10,22,0.82));
      border: 1px solid rgba(148, 163, 184, 0.10);
    }}
    svg text {{ fill: var(--muted); font-family: inherit; font-size: 12px; }}
    table {{ width: 100%; border-collapse: collapse; overflow: hidden; border-radius: 16px; }}
    th, td {{ padding: 12px 14px; border-bottom: 1px solid var(--line); text-align: right; font-size: 13px; }}
    th {{ color: #cbd5e1; font-weight: 700; background: rgba(96, 165, 250, 0.075); position: sticky; top: 0; }}
    td:first-child, th:first-child {{ text-align: left; }}
    tbody tr:nth-child(even) {{ background: rgba(255,255,255,0.025); }}
    tbody tr:hover {{ background: rgba(96, 165, 250, 0.07); }}
    tr:last-child td {{ border-bottom: none; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 18px; max-height: 560px; }}
    .config {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }}
    .config div {{ padding: 13px 14px; border-radius: 16px; background: rgba(255,255,255,0.045); border: 1px solid rgba(148,163,184,0.08); }}
    .config span {{ display: block; color: var(--muted); font-size: 12px; margin-bottom: 5px; }}
    .config strong {{ font-size: 15px; }}
    .note {{ padding: 16px 18px; border-radius: 18px; background: rgba(251, 191, 36, 0.1); border: 1px solid rgba(251, 191, 36, 0.24); }}
    .filters {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 0 0 14px; }}
    .filter-btn {{
      border: 1px solid var(--line); color: var(--muted); background: rgba(255,255,255,0.045);
      border-radius: 999px; padding: 8px 13px; cursor: pointer; font: inherit; font-size: 13px;
    }}
    .filter-btn:hover {{ border-color: rgba(96, 165, 250, 0.45); color: var(--text); }}
    .filter-btn.active {{ color: var(--text); background: rgba(96, 165, 250, 0.22); border-color: rgba(96, 165, 250, 0.62); }}
    footer {{ margin-top: 24px; color: var(--muted); font-size: 12px; }}
    @media (max-width: 980px) {{
      header {{ display: block; }}
      .cards, .two, .config {{ grid-template-columns: 1fr; }}
      main {{ width: min(100vw - 24px, 1280px); padding-top: 22px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>{title}</h1>
        <p>{subtitle}</p>
      </div>
      <div class="badge">Сгенерировано: {generated_at}</div>
    </header>

    <section class="grid cards">
      {metric_cards}
    </section>

    <section class="panel section">
      <h2>Настройки лейблинга</h2>
      <div class="config">{config_items}</div>
    </section>

    <section class="grid two section">
      <div class="panel">
        <h2>Распределение Target</h2>
        <div class="chart">{target_distribution_chart}</div>
      </div>
      <div class="panel">
        <h2>Стабильность покрытия по месяцам</h2>
        <div class="chart">{monthly_coverage_chart}</div>
      </div>
    </section>

    <section class="panel section">
      <h2>Итоги стабильности</h2>
      {stability_summary}
    </section>

    <section class="grid two section">
      <div class="panel">
        <h2>Качество исходов и net PnL</h2>
        <div class="table-wrap">{outcome_quality_table}</div>
      </div>
      <div class="panel">
        <h2>Пересечение Long / Short labels</h2>
        <div class="table-wrap">{overlap_table}</div>
      </div>
    </section>

    <section class="grid two section">
      <div class="panel">
        <h2>Зависимость от adaptive horizon</h2>
        <div class="table-wrap">{horizon_quality_table}</div>
      </div>
      <div class="panel">
        <h2>Распределение барьеров и horizon</h2>
        <div class="table-wrap">{barrier_summary_table}</div>
      </div>
    </section>

    <section class="grid two section">
      <div class="panel">
        <h2>Самые слабые месяцы</h2>
        <div class="table-wrap">{weakest_months_table}</div>
      </div>
      <div class="panel">
        <h2>Самые сильные месяцы</h2>
        <div class="table-wrap">{strongest_months_table}</div>
      </div>
    </section>

    <section class="panel section">
      <h2>Покрытие по символам</h2>
      <div class="table-wrap">{by_symbol_table}</div>
    </section>

    <section class="panel section">
      <h2>Качество labels по символам</h2>
      <div class="table-wrap">{by_symbol_quality_table}</div>
    </section>

    <section class="panel section">
      <h2>Оценка LSTM sequence samples</h2>
      <div class="table-wrap">{sequence_samples_table}</div>
    </section>

    <section class="panel section">
      <h2>Помесячное покрытие</h2>
      <div class="filters" id="monthly-year-filters">{monthly_year_filters}</div>
      <div class="table-wrap">{monthly_table}</div>
    </section>

    <section class="panel section">
      <h2>Помесячное качество labels и PnL</h2>
      <div class="table-wrap">{monthly_quality_table}</div>
    </section>

    <section class="note section">
      <p>{interpretation_note}</p>
    </section>

    <footer>
      Report source: label_coverage_report.py
    </footer>
  </main>
  <script>
    (function () {{
      const buttons = Array.from(document.querySelectorAll("[data-year-filter]"));
      const rows = Array.from(document.querySelectorAll("#monthly-coverage-table tbody tr"));
      function applyFilter(year) {{
        buttons.forEach((button) => button.classList.toggle("active", button.dataset.yearFilter === year));
        rows.forEach((row) => {{
          row.style.display = year === "all" || row.dataset.year === year ? "" : "none";
        }});
      }}
      buttons.forEach((button) => button.addEventListener("click", () => applyFilter(button.dataset.yearFilter)));
      if (buttons.length > 0) {{
        applyFilter(buttons[0].dataset.yearFilter);
      }}
    }})();
  </script>
</body>
</html>
"""
