@app.post("/analyze_csv")
def analyze_csv(
    file: UploadFile = File(...),
    pair: Optional[str] = Form(None),
    timeframe: Optional[str] = Form(None),
    auto_backtest: Optional[str] = Form("true"),
    auto_log: Optional[str] = Form("true")
):
    """
    Analisis file CSV candlestick (open, high, low, close)
    dan otomatis simpan hasil ke trade_log.csv + retrain model jika perlu.
    """
    auto_bt = auto_backtest.lower() != "false"
    auto_lg = auto_log.lower() != "false"

    try:
        contents = file.file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid_csv: {e}")

    # pastikan kolom valid
    df.columns = [c.strip().lower() for c in df.columns]
    def find_col(k): return next((c for c in df.columns if k in c), None)
    o, h, l, ccol = find_col('open'), find_col('high'), find_col('low'), find_col('close')
    if not all([o, h, l, ccol]):
        raise HTTPException(status_code=400, detail="kolom_tidak_lengkap (butuh open, high, low, close)")

    # ubah kolom ke format standar
    df2 = df[[o, h, l, ccol]].rename(columns={o: 'open', h: 'high', l: 'low', ccol: 'close'})
    for col in ['open', 'high', 'low', 'close']:
        df2[col] = pd.to_numeric(df2[col], errors='coerce')
    df2 = df2.dropna().reset_index(drop=True)

    # analisis teknikal
    res = hybrid_analyze(df2, pair=pair or "CSV", timeframe=timeframe or "csv")
    res = _postprocess_with_learning(res)

    # backtest otomatis (kalau BACKTEST_URL aktif)
    bt_res = {}
    if auto_bt and res.get("signal_type") != "WAIT":
        bt_payload = {
            "pair": res["pair"], "timeframe": res["timeframe"], "side": res["signal_type"],
            "entry": res["entry"], "tp1": res.get("tp1"), "tp2": res.get("tp2"),
            "sl": res["sl"], "confidence": res["confidence"], "reason": res["reasoning"]
        }
        bt_res = post_to_backtester(bt_payload)
        res["backtest_raw"] = bt_res

    # simpan otomatis ke trade_log.csv
    if auto_lg:
        append_trade_log({
            "pair": res["pair"],
            "timeframe": res["timeframe"],
            "signal_type": res["signal_type"],
            "entry": res["entry"],
            "tp1": res["tp1"],
            "tp2": res["tp2"],
            "sl": res["sl"],
            "confidence": res["confidence"],
            "reasoning": res["reasoning"],
            "backtest_hit": bt_res.get("hit") if isinstance(bt_res, dict) else None,
            "backtest_pnl": bt_res.get("pnl_total") if isinstance(bt_res, dict) else None
        })

        # retrain otomatis kalau cukup sinyal
        check_and_trigger_retrain_if_needed()

    # info hasil
    res["bars_used"] = int(df2.shape[0])
    res["auto_logged"] = auto_lg
    res["auto_retrain_triggered"] = auto_lg
    return respond(res)
