import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ── SAYFA AYARLARI ───────────────────────────────────────────────────
st.set_page_config(
    page_title="BIST30 Karar Destek Sistemi",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Genel arka plan */
[data-testid="stAppViewContainer"] {
    background: #F8F9FB;
}
[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E8EBF0;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: #6B7280;
    font-size: 12px;
}

/* Başlık renkleri */
h1 { color: #111827 !important; font-size: 22px !important; }
h2 { color: #374151 !important; font-size: 16px !important; }
h3 { color: #374151 !important; font-size: 14px !important; }

/* Buton */
.stButton > button {
    background: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    width: 100% !important;
    padding: 0.65rem 1rem !important;
    font-size: 14px !important;
    transition: background .15s !important;
}
.stButton > button:hover {
    background: #1D4ED8 !important;
}

/* Tüm arka planları zorla açık yap */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"], .main, .block-container {
    background-color: #F8F9FB !important;
    color: #111827 !important;
}

/* Sidebar arka plan */
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {
    background-color: #FFFFFF !important;
}

/* Tüm yazı renkleri */
p, span, label, div, h1, h2, h3, h4 {
    color: #111827 !important;
}

/* Slider track */
[data-testid="stSlider"] > div > div > div {
    background: #E8EBF0 !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #2563EB !important;
    border-color: #2563EB !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[style] {
    background: #2563EB !important;
}

/* Slider value box */
[data-testid="stSlider"] div[data-baseweb="input"] {
    background: #FFFFFF !important;
    border-color: #E8EBF0 !important;
}
[data-testid="stSlider"] input {
    background: #FFFFFF !important;
    color: #111827 !important;
}

/* Text input */
[data-testid="stTextInput"] input {
    background: #FFFFFF !important;
    color: #111827 !important;
    border-color: #E8EBF0 !important;
}

/* Radio butonlar */
[data-testid="stRadio"] label {
    color: #374151 !important;
}

/* Date input */
[data-testid="stDateInput"] input {
    background: #FFFFFF !important;
    color: #111827 !important;
}

/* Genel kutu ve kartlar */
[data-baseweb="base-input"] {
    background: #FFFFFF !important;
}
[data-baseweb="select"] {
    background: #FFFFFF !important;
}

/* Metrik kartları */
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E8EBF0;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}
.kpi-label {
    font-size: 11px;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}
.kpi-value {
    font-size: 20px;
    font-weight: 600;
    margin-top: 2px;
}
.kpi-green  { color: #059669; }
.kpi-red    { color: #DC2626; }
.kpi-blue   { color: #2563EB; }
.kpi-gray   { color: #6B7280; }

/* Kazanan banner */
.winner-banner {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-left: 4px solid #2563EB;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 1.5rem;
}
.winner-banner.near {
    background: #FFFBEB;
    border-color: #FDE68A;
    border-left-color: #F59E0B;
}
.winner-badge {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #2563EB;
    margin-bottom: 4px;
}
.winner-badge.near { color: #D97706; }
.winner-title {
    font-size: 22px;
    font-weight: 700;
    color: #111827;
    margin-bottom: 4px;
}
.winner-meta {
    font-size: 13px;
    color: #6B7280;
}
.winner-meta strong { color: #2563EB; }

/* Tablo */
.stDataFrame {
    border: 1px solid #E8EBF0;
    border-radius: 10px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)


# ── FONKSİYONLAR ─────────────────────────────────────────────────────

def hex_to_rgba(hex_color, alpha=0.1):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

@st.cache_data(show_spinner=False)
def get_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.columns = [col.strip() for col in df.columns]
    return df.dropna()

def add_indicators(df):
    delta = df['Close'].diff()
    gain  = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss  = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    ema12        = df['Close'].ewm(span=12, adjust=False).mean()
    ema26        = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']   = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['MA21']  = df['Close'].rolling(21).mean()
    df['Std']   = df['Close'].rolling(21).std()
    df['Upper'] = df['MA21'] + 2 * df['Std']
    df['Lower'] = df['MA21'] - 2 * df['Std']

    df['SMA50']  = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()

    low14        = df['Low'].rolling(14).min()
    high14       = df['High'].rolling(14).max()
    df['perc_K'] = (df['Close'] - low14) / (high14 - low14 + 1e-10) * 100
    df['perc_D'] = df['perc_K'].rolling(3).mean()

    df['OBV']    = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV_MA'] = df['OBV'].rolling(20).mean()
    return df

def apply_strategies(df):
    df['S_BH']    = 1
    df['S_RSI50'] = np.where(df['RSI'] > 50, 1, 0)
    df['S_MACD']  = np.where(df['MACD'] > df['Signal'], 1, 0)

    df['S_BB'] = np.nan
    df.loc[df['Close'] < df['Lower'], 'S_BB'] = 1
    df.loc[df['Close'] > df['Upper'], 'S_BB'] = 0
    df['S_BB'] = df['S_BB'].ffill().fillna(0)

    df['S_SMA']  = np.where(df['SMA50'] > df['SMA200'], 1, 0)
    df['S_STOK'] = np.where(df['perc_K'] > df['perc_D'], 1, 0)
    df['S_OBV']  = np.where(df['OBV'] > df['OBV_MA'], 1, 0)
    return df

def run_backtest(df, strategy_col):
    dc              = df.copy()
    dc['Ret']       = dc['Close'].pct_change()
    dc['Strat_Ret'] = dc[strategy_col].shift(1) * dc['Ret']

    equity       = (1 + dc['Strat_Ret'].fillna(0)).cumprod()
    total_return = equity.iloc[-1]
    years        = (dc.index[-1] - dc.index[0]).days / 365.25
    cagr         = (total_return ** (1 / years)) - 1 if years > 0 else 0

    dd     = (equity - equity.cummax()) / equity.cummax()
    max_dd = dd.min()

    neg_days     = dc['Strat_Ret'][dc['Strat_Ret'] < 0]
    downside_std = neg_days.std() * np.sqrt(252) if len(neg_days) > 0 else np.nan
    sortino = cagr / downside_std if (
        downside_std and not np.isnan(downside_std) and downside_std != 0
    ) else None
    calmar = cagr / abs(max_dd) if max_dd != 0 else None

    if strategy_col == 'S_BH':
        return {
            'cagr': round(cagr * 100, 2),
            'total_return': round(total_return, 2),
            'max_dd': round(max_dd * 100, 2),
            'sortino': round(sortino, 2) if sortino else None,
            'calmar':  round(calmar,  2) if calmar  else None,
            'win_rate': None,
            'profit_factor': None,
            'trades': 0,
            'equity': equity
        }

    signals      = dc[strategy_col].shift(1).fillna(0)
    prices       = dc['Close']
    wins, losses = [], []
    entry_price, in_pos = 0, False

    for i in range(1, len(dc)):
        if signals.iloc[i] == 1 and signals.iloc[i-1] == 0:
            entry_price = prices.iloc[i]
            in_pos = True
        elif signals.iloc[i] == 0 and signals.iloc[i-1] == 1 and in_pos:
            pnl = prices.iloc[i] - entry_price
            (wins if pnl > 0 else losses).append(abs(pnl))
            in_pos = False

    trade_count   = len(wins) + len(losses)
    win_rate      = len(wins) / trade_count * 100 if trade_count > 0 else 0
    profit_factor = sum(wins) / sum(losses) if sum(losses) > 0 else None

    return {
        'cagr': round(cagr * 100, 2),
        'total_return': round(total_return, 2),
        'max_dd': round(max_dd * 100, 2),
        'sortino': round(sortino, 2) if sortino else None,
        'calmar':  round(calmar,  2) if calmar  else None,
        'win_rate': round(win_rate, 1),
        'profit_factor': round(profit_factor, 2) if profit_factor else None,
        'trades': trade_count,
        'equity': equity
    }


# ── SIDEBAR ──────────────────────────────────────────────────────────
# ── BIST30 HİSSE LİSTESİ ────────────────────────────────────────────
BIST30_HISSELER = {
    "AEFES.IS":  "Anadolu Efes",
    "AKBNK.IS":  "Akbank",
    "ASELS.IS":  "Aselsan",
    "ASTOR.IS":  "Astor Enerji",
    "BIMAS.IS":  "BİM Mağazalar",
    "DSTKF.IS":  "Destek Faktoring",
    "EKGYO.IS":  "Emlak Konut GYO",
    "ENKAI.IS":  "Enka İnşaat",
    "EREGL.IS":  "Ereğli Demir Çelik",
    "FROTO.IS":  "Ford Otosan",
    "GARAN.IS":  "Garanti BBVA",
    "GUBRF.IS":  "Gübre Fabrikaları",
    "ISCTR.IS":  "İş Bankası",
    "KCHOL.IS":  "Koç Holding",
    "KRDMD.IS":  "Kardemir",
    "MGROS.IS":  "Migros",
    "PETKM.IS":  "Petkim",
    "PGSUS.IS":  "Pegasus",
    "SAHOL.IS":  "Sabancı Holding",
    "SASA.IS":   "SASA Polyester",
    "SISE.IS":   "Şişe Cam",
    "TAVHL.IS":  "TAV Havalimanları",
    "TCELL.IS":  "Turkcell",
    "THYAO.IS":  "Türk Hava Yolları",
    "TOASO.IS":  "Tofaş Oto",
    "TRALT.IS":  "Türkiye Alüminyum",
    "TTKOM.IS":  "Türk Telekom",
    "TUPRS.IS":  "Tüpraş",
    "VAKBN.IS":  "Vakıfbank",
    "YKBNK.IS":  "Yapı Kredi",
}

with st.sidebar:
    st.markdown("## 📊 BIST30 Karar Destek")
    st.markdown("---")

    st.markdown("### Hisse & Dönem")

    # Arama kutusu
    search_query = st.text_input(
        "Hisse ara",
        value="",
        placeholder="Örn: T yazınca THYAO, TUPRS...",
        help="Hisse kodu veya şirket adından arama yap"
    )

    # Filtrele
    if search_query:
        filtered = {
            k: v for k, v in BIST30_HISSELER.items()
            if search_query.upper() in k or search_query.upper() in v.upper()
        }
    else:
        filtered = BIST30_HISSELER

    if filtered:
        secenekler = [f"{k} — {v}" for k, v in filtered.items()]
        secim = st.selectbox("Hisse seç", secenekler)
        ticker = secim.split(" — ")[0]
        st.caption(f"Seçili: **{ticker}** — {BIST30_HISSELER.get(ticker, '')}")
    else:
        # Listede yoksa kullanıcının yazdığını direkt ticker olarak kullan
        manual = search_query.strip().upper()
        if not manual.endswith(".IS"):
            manual = manual + ".IS"
        ticker = manual
        st.info(f"'{ticker}' listede yok — direkt Yahoo Finance'den çekilecek.")
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Başlangıç", value=pd.to_datetime("2021-01-01"))
    with c2:
        end_date = st.date_input("Bitiş", value=pd.to_datetime("2026-04-20"))

    st.markdown("---")
    st.markdown("### Risk Toleransı")

    max_dd_tol = st.slider(
        "Maks. Drawdown (%)",
        min_value=-80, max_value=-5, value=-40, step=1,
        help="En kötü anda portföyün yüzde kaç düşmesine katlanabilirsin?"
    )
    min_cagr = st.slider(
        "Min. CAGR (%)",
        min_value=-20, max_value=80, value=10, step=1,
        help="Yıllık minimum bileşik büyüme oranı"
    )
    min_sortino = st.slider(
        "Min. Sortino",
        min_value=-1.0, max_value=3.0, value=0.5, step=0.1,
        help="Sadece aşağı yönlü risk bazlı verimlilik ölçütü"
    )
    min_calmar = st.slider(
        "Min. Calmar",
        min_value=0.0, max_value=2.0, value=0.3, step=0.1,
        help="CAGR / Max Drawdown — en kötü senaryoya değdi mi?"
    )
    min_win_rate = st.slider(
        "Min. Win Rate (%)",
        min_value=0, max_value=80, value=40, step=1,
        help="Kârlı kapanan işlemlerin yüzdesi"
    )
    min_pf = st.slider(
        "Min. Profit Factor",
        min_value=0.5, max_value=3.0, value=1.2, step=0.1,
        help="Toplam kazanç / Toplam kayıp — 1.0 altı her zaman zararlı"
    )
    invest_years = st.slider(
        "Yatırım süresi (yıl)",
        min_value=1, max_value=20, value=5, step=1,
        help="Paranı kaç yıl tutmayı planlıyorsun?"
    )

    st.markdown("---")
    st.markdown("### Öncelik Profili")
    priority = st.radio(
        "",
        ["Dengeli", "Büyüme odaklı", "Güvenlik odaklı", "Verimlilik odaklı"],
        index=0
    )

    st.markdown("---")
    run_btn = st.button("🚀  Analizi Çalıştır")


# ── ANA ALAN ─────────────────────────────────────────────────────────
st.markdown("# 📊 BIST30 Algoritmik Strateji Karar Destek Sistemi")
st.markdown(
    "Teknik analiz stratejilerini çok boyutlu metriklerle karşılaştır, "
    "risk profiline göre en uygun stratejiyi bul."
)

STRATS = {
    'Al-Tut (Piyasa)': 'S_BH',
    'RSI (50)':         'S_RSI50',
    'MACD':             'S_MACD',
    'Bollinger':        'S_BB',
    'SMA (50/200)':     'S_SMA',
    'Stokastik':        'S_STOK',
    'OBV (Hacim)':      'S_OBV',
}

WEIGHTS = {
    "Dengeli":           {'cagr':.20,'sortino':.25,'calmar':.20,'dd':.15,'wr':.10,'pf':.10},
    "Büyüme odaklı":     {'cagr':.45,'sortino':.15,'calmar':.10,'dd':.10,'wr':.10,'pf':.10},
    "Güvenlik odaklı":   {'cagr':.10,'sortino':.20,'calmar':.30,'dd':.25,'wr':.10,'pf':.05},
    "Verimlilik odaklı": {'cagr':.15,'sortino':.35,'calmar':.20,'dd':.10,'wr':.10,'pf':.10},
}

COLORS = ['#2563EB','#059669','#D97706','#DC2626','#7C3AED','#DB2777','#0891B2']

if run_btn:
    with st.spinner(f"{ticker} verisi çekiliyor ve analiz yapılıyor..."):
        try:
            data = get_data(ticker, str(start_date), str(end_date))
            if data.empty:
                st.error("Veri çekilemedi. Hisse kodunu kontrol et. Örnek: SASA.IS")
                st.stop()

            data    = add_indicators(data)
            data    = apply_strategies(data)
            results = {name: run_backtest(data, col) for name, col in STRATS.items()}

            # ── SKORLAMA ─────────────────────────────────────────────
            w = WEIGHTS[priority]

            def norm(v, mn, mx):
                return max(0.0, min(1.0, (v - mn) / (mx - mn))) if mx > mn else 0.0

            cagr_v = [r['cagr']    for r in results.values()]
            so_v   = [r['sortino'] for r in results.values() if r['sortino'] is not None]
            ca_v   = [r['calmar']  for r in results.values() if r['calmar']  is not None]
            dd_v   = [r['max_dd']  for r in results.values()]
            wr_v   = [r['win_rate']      for r in results.values() if r['win_rate']      is not None]
            pf_v   = [r['profit_factor'] for r in results.values() if r['profit_factor'] is not None]

            scored = []
            for name, m in results.items():
                nC  = norm(m['cagr'],         min(cagr_v), max(cagr_v))
                nS  = norm(m['sortino'] or 0, min(so_v or [0]), max(so_v or [1]))
                nCa = norm(m['calmar']  or 0, min(ca_v or [0]), max(ca_v or [1]))
                nD  = norm(-m['max_dd'],      -max(dd_v), -min(dd_v))
                nW  = norm(m['win_rate']      or 0, min(wr_v or [0]), max(wr_v or [1]))
                nP  = norm(m['profit_factor'] or 0, min(pf_v or [0]), max(pf_v or [1]))

                score = round(
                    (w['cagr']*nC + w['sortino']*nS + w['calmar']*nCa +
                     w['dd']*nD + w['wr']*nW + w['pf']*nP) * 100
                )

                passes = (
                    m['max_dd'] >= max_dd_tol and
                    m['cagr']   >= min_cagr and
                    (m['sortino'] is None or m['sortino'] >= min_sortino) and
                    (m['calmar']  is None or m['calmar']  >= min_calmar) and
                    (m['win_rate']      is None or m['win_rate']      >= min_win_rate) and
                    (m['profit_factor'] is None or m['profit_factor'] >= min_pf)
                )

                tot = round((1 + m['cagr'] / 100) ** invest_years, 2)
                scored.append({
                    'name': name, 'score': score, 'passes': passes, 'tot': tot,
                    'nC': nC, 'nS': nS, 'nCa': nCa, 'nD': nD, 'nW': nW, 'nP': nP,
                    **m
                })

            scored.sort(key=lambda x: (x['passes'], x['score']), reverse=True)
            winner = scored[0]

            # ── KAZANAN BANNER ────────────────────────────────────────
            banner_class = "winner-banner" if winner['passes'] else "winner-banner near"
            badge_class  = "winner-badge"  if winner['passes'] else "winner-badge near"
            badge_text   = "✓ Tüm kriterler karşılandı" if winner['passes'] else "~ En yakın strateji"

            st.markdown(f"""
            <div class="{banner_class}">
              <div class="{badge_class}">{badge_text}</div>
              <div class="winner-title">{winner['name']}</div>
              <div class="winner-meta">
                Uyum skoru: <strong>{winner['score']}/100</strong>
                &nbsp;·&nbsp;
                {invest_years} yılda toplam getiri: <strong>{winner['tot']}x</strong>
                &nbsp;·&nbsp;
                Öncelik: <strong>{priority}</strong>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── METRİK KARTLARI ───────────────────────────────────────
            def color(val, threshold, higher_is_better=True):
                if val is None: return "kpi-gray"
                ok = val >= threshold if higher_is_better else val <= threshold
                return "kpi-green" if ok else "kpi-red"

            cols = st.columns(8)
            cards = [
                ("CAGR", f"%{winner['cagr']}",
                 color(winner['cagr'], min_cagr)),
                (f"{invest_years}Y Getiri", f"{winner['tot']}x",
                 "kpi-blue"),
                ("Max DD", f"%{winner['max_dd']}",
                 color(winner['max_dd'], max_dd_tol, False)),
                ("Sortino", str(winner['sortino']) if winner['sortino'] else "N/A",
                 color(winner['sortino'], min_sortino) if winner['sortino'] else "kpi-gray"),
                ("Calmar", str(winner['calmar']) if winner['calmar'] else "N/A",
                 color(winner['calmar'], min_calmar) if winner['calmar'] else "kpi-gray"),
                ("Win Rate", f"%{winner['win_rate']}" if winner['win_rate'] else "N/A",
                 color(winner['win_rate'], min_win_rate) if winner['win_rate'] else "kpi-gray"),
                ("Profit F.", str(winner['profit_factor']) if winner['profit_factor'] else "N/A",
                 color(winner['profit_factor'], min_pf) if winner['profit_factor'] else "kpi-gray"),
                ("İşlem", str(winner['trades']),
                 "kpi-gray"),
            ]
            for col, (label, val, cls) in zip(cols, cards):
                col.markdown(f"""
                <div class="kpi-card">
                  <div class="kpi-label">{label}</div>
                  <div class="kpi-value {cls}">{val}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── SEKMELER ─────────────────────────────────────────────
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Equity Eğrileri",
                "📊 Performans Matrisi",
                "🔭 Radar Grafiği",
                "🏆 Sıralama"
            ])

            with tab1:
                fig = go.Figure()
                for i, (name, m) in enumerate(results.items()):
                    is_winner = name == winner['name']
                    fig.add_trace(go.Scatter(
                        x=m['equity'].index,
                        y=m['equity'].values,
                        name=name,
                        line=dict(
                            color=COLORS[i % len(COLORS)],
                            width=3 if is_winner else 1.5,
                            dash='solid' if is_winner else 'dot'
                        ),
                        opacity=1.0 if is_winner else 0.55
                    ))
                fig.update_layout(
                    paper_bgcolor='#FFFFFF',
                    plot_bgcolor='#FAFAFA',
                    font=dict(color='#374151', family='sans-serif'),
                    legend=dict(
                        bgcolor='#FFFFFF',
                        bordercolor='#E8EBF0',
                        borderwidth=1,
                        font=dict(color='#374151', size=12)
                    ),
                    xaxis=dict(
                        gridcolor='#F3F4F6',
                        linecolor='#E8EBF0',
                        color='#9CA3AF'
                    ),
                    yaxis=dict(
                        gridcolor='#F3F4F6',
                        linecolor='#E8EBF0',
                        color='#9CA3AF',
                        title='Portföy Değeri (x)'
                    ),
                    margin=dict(l=0, r=0, t=20, b=0),
                    height=420
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                rows = []
                for s in scored:
                    rows.append({
                        'Strateji':             s['name'],
                        'CAGR (%)':             s['cagr'],
                        f'{invest_years}Y (x)': s['tot'],
                        'Max DD (%)':           s['max_dd'],
                        'Sortino':              s['sortino'],
                        'Calmar':               s['calmar'],
                        'Win Rate (%)':         s['win_rate'],
                        'Profit Factor':        s['profit_factor'],
                        'İşlem':                s['trades'],
                        'Skor':                 s['score'],
                    })
                df_rep = pd.DataFrame(rows).set_index('Strateji')
                st.dataframe(df_rep, use_container_width=True, height=310)

            with tab3:
                top4       = scored[:4]
                categories = ['CAGR', 'Sortino', 'Calmar', 'Güvenlik', 'Win Rate', 'Profit F.']
                fig_r      = go.Figure()
                for i, s in enumerate(top4):
                    vals = [
                        round(s['nC']  * 100),
                        round(s['nS']  * 100),
                        round(s['nCa'] * 100),
                        round(s['nD']  * 100),
                        round(s['nW']  * 100),
                        round(s['nP']  * 100),
                    ]
                    fig_r.add_trace(go.Scatterpolar(
                        r=vals + [vals[0]],
                        theta=categories + [categories[0]],
                        name=s['name'],
                        line=dict(color=COLORS[i], width=2.5 if i == 0 else 1.5),
                        fill='toself',
                        fillcolor=hex_to_rgba(COLORS[i], 0.08),
                    ))
                fig_r.update_layout(
                    polar=dict(
                        bgcolor='#FAFAFA',
                        radialaxis=dict(
                            visible=True, range=[0, 100],
                            gridcolor='#E8EBF0',
                            tickfont=dict(color='#9CA3AF', size=10)
                        ),
                        angularaxis=dict(
                            gridcolor='#E8EBF0',
                            tickfont=dict(color='#374151', size=12)
                        )
                    ),
                    paper_bgcolor='#FFFFFF',
                    font=dict(color='#374151'),
                    legend=dict(
                        bgcolor='#FFFFFF',
                        bordercolor='#E8EBF0',
                        borderwidth=1,
                        font=dict(color='#374151', size=12)
                    ),
                    height=460,
                    margin=dict(l=60, r=60, t=30, b=30)
                )
                st.plotly_chart(fig_r, use_container_width=True)

            with tab4:
                for i, s in enumerate(scored):
                    icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
                    dot  = "🟢" if s['passes'] else "⚪"
                    st.markdown(
                        f"**{icon} {s['name']}** {dot} &nbsp;"
                        f"Skor: `{s['score']}/100` &nbsp;·&nbsp;"
                        f"CAGR: `%{s['cagr']}` &nbsp;·&nbsp;"
                        f"{invest_years}Y: `{s['tot']}x` &nbsp;·&nbsp;"
                        f"Max DD: `%{s['max_dd']}`"
                    )
                    st.progress(s['score'] / 100)
                    st.markdown("")

        except Exception as e:
            st.error(f"Hata oluştu: {e}")
            st.exception(e)

else:
    # ── KARŞILAMA EKRANI ─────────────────────────────────────────────
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 📌 Nasıl çalışır?")
        st.markdown("""
        1. Sol panelden **hisse kodunu** gir
        2. **Risk toleransını** slider'larla ayarla
        3. **Öncelik profilini** seç
        4. **Analizi Çalıştır** butonuna bas
        """)

    with col2:
        st.markdown("#### 📐 Ölçülen metrikler")
        st.markdown("""
        - **CAGR** — Yıllık bileşik büyüme
        - **Max Drawdown** — En kötü düşüş
        - **Sortino** — Aşağı yönlü risk/getiri
        - **Calmar** — En kötü senaryo dayanıklılığı
        - **Win Rate** — İsabet oranı
        - **Profit Factor** — Kazanç/kayıp oranı
        """)

    with col3:
        st.markdown("#### 💡 Örnek hisse kodları")
        st.markdown("""
        - `SASA.IS` — SASA Polyester
        - `THYAO.IS` — Türk Hava Yolları
        - `AKBNK.IS` — Akbank
        - `GARAN.IS` — Garanti BBVA
        - `TUPRS.IS` — Tüpraş
        - `EREGL.IS` — Ereğli Demir Çelik
        """)

    st.markdown("---")
    st.info("👈 Sol panelden başlayın — hisse kodunu girin ve **Analizi Çalıştır** butonuna basın.")
