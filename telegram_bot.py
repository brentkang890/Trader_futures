# telegram_bot.py
import os
import time
import requests

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
APP_URL = os.environ.get("APP_URL", "https://web-production-af34.up.railway.app")

if not BOT_TOKEN or not CHAT_ID:
    raise ValueError("Environment variable BOT_TOKEN dan CHAT_ID harus diatur.")

BASE = f"https://api.telegram.org/bot{BOT_TOKEN}"

def send_message(text, parse_mode="HTML"):
    try:
        url = f"{BASE}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": text[:4096], "parse_mode": parse_mode}
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print("[ERROR] send_message:", e)

def get_updates(offset=None, timeout=60):
    try:
        url = f"{BASE}/getUpdates"
        params = {"timeout": timeout, "offset": offset}
        r = requests.get(url, params=params, timeout=timeout+10)
        return r.json()
    except Exception as e:
        print("[ERROR] get_updates:", e)
        return {}

def handle_command(text):
    if not text:
        return "⚠️ Pesan kosong."
    text = text.strip()
    lower = text.lower()
    if lower == "status":
        try:
            r = requests.get(f"{APP_URL}/learning_status", timeout=10); d = r.json()
            return ("🤖 <b>Status Model AI</b>\n"
                    f"📦 Model: {'✅ Ada' if d.get('model_exists') else '❌ Tidak ada'}\n"
                    f"📊 Data Log: {d.get('trade_log_count', 0)} sinyal")
        except Exception as e:
            return f"⚠️ Gagal ambil status: {e}"
    if lower == "log":
        try:
            r = requests.get(f"{APP_URL}/logs_summary", timeout=10); d = r.json()
            if "detail" in d:
                return d["detail"]
            return (f"📊 <b>{d.get('pair')}</b> ({d.get('timeframe')})\n"
                    f"💡 Signal: <b>{d.get('signal_type')}</b>\n"
                    f"🎯 Entry: {d.get('entry')}\n"
                    f"TP1: {d.get('tp1')} | TP2: {d.get('tp2')}\n"
                    f"🛡 SL: {d.get('sl')}\n"
                    f"📈 Confidence: {d.get('confidence')}\n"
                    f"🧠 {d.get('reasoning')}")
        except Exception as e:
            return f"⚠️ Gagal ambil log: {e}"
    if lower.startswith("scalp "):
        parts = text.split()
        if len(parts) < 2:
            return "Format scalp salah. Contoh: <code>scalp BTCUSDT</code>"
        pair = parts[1].upper()
        try:
            r = requests.get(f"{APP_URL}/scalp_signal?pair={pair}&tf=3m&auto_log=true", timeout=20); d = r.json()
            if "error" in d:
                return f"⚠️ {d['error']}"
            return (f"⚡️ <b>Scalp {d.get('pair')}</b> ({d.get('timeframe')})\n"
                    f"💡 Signal: <b>{d.get('signal_type')}</b>\n"
                    f"🎯 Entry: {d.get('entry')}\n"
                    f"TP1: {d.get('tp1')} | TP2: {d.get('tp2')}\n"
                    f"🛡 SL: {d.get('sl')}\n📈 Confidence: {d.get('confidence')}")
        except Exception as e:
            return f"⚠️ Gagal ambil scalp: {e}"
    # Default: treat as pair + optional timeframe (e.g. "BTCUSDT 15m")
    parts = text.split()
    if len(parts) == 0:
        return "⚠️ Format salah. Contoh: <code>BTCUSDT 15m</code>"
    pair = parts[0].upper()
    tf = parts[1] if len(parts) > 1 else "15m"
    try:
        r = requests.get(f"{APP_URL}/pro_signal?pair={pair}&tf_main=1h&tf_entry={tf}&auto_log=true", timeout=25)
        d = r.json()
    except Exception as e:
        return f"❌ Error: {e}"
    if "error" in d:
        return f"⚠️ {d['error']}"
    return (f"📊 <b>{d.get('pair')}</b> ({d.get('timeframe')})\n"
            f"💡 Signal: <b>{d.get('signal_type')}</b>\n"
            f"🎯 Entry: {d.get('entry')}\n"
            f"TP1: {d.get('tp1')} | TP2: {d.get('tp2')}\n"
            f"🛡 SL: {d.get('sl')}\n"
            f"📈 Confidence: {d.get('confidence')}\n"
            f"🧠 {d.get('reasoning')}")

def main():
    offset = None
    send_message("🤖 ProTraderAI Bot aktif.")
    while True:
        try:
            updates = get_updates(offset, timeout=60)
            if "result" in updates:
                for u in updates["result"]:
                    offset = u["update_id"] + 1
                    msg = u.get("message") or u.get("edited_message") or {}
                    if not msg:
                        continue
                    if "text" in msg:
                        reply = handle_command(msg["text"])
                        send_message(reply)
                    elif "photo" in msg:
                        # take highest res
                        photo = msg["photo"][-1]
                        file_id = photo["file_id"]
                        file_info = requests.get(f"{BASE}/getFile", params={"file_id": file_id}, timeout=10).json()
                        file_path = file_info.get("result", {}).get("file_path")
                        if not file_path:
                            send_message("⚠️ Gagal ambil file photo.")
                            continue
                        file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
                        img = requests.get(file_url, timeout=20).content
                        send_message("📷 Menganalisis gambar chart, mohon tunggu...")
                        try:
                            resp = requests.post(f"{APP_URL}/analyze_chart", files={"file": ("chart.jpg", img)}, timeout=60)
                            if resp.status_code == 200:
                                d = resp.json()
                                caption = (f"📊 <b>{d.get('pair')}</b> ({d.get('timeframe')})\n"
                                           f"💡 Signal: <b>{d.get('signal_type')}</b>\n"
                                           f"🎯 Entry: {d.get('entry')}\n"
                                           f"TP1: {d.get('tp1')} | TP2: {d.get('tp2')}\n"
                                           f"🛡 SL: {d.get('sl')}\n"
                                           f"📈 Confidence: {d.get('confidence')}\n"
                                           f"🧠 {d.get('reasoning')}")
                                send_message(caption)
                            else:
                                send_message(f"⚠️ Gagal analisis gambar: {resp.status_code} {resp.text[:200]}")
                        except Exception as e:
                            send_message(f"⚠️ Error saat analisis gambar: {e}")
            time.sleep(1)
        except Exception as e:
            print("[LOOP ERROR]", e)
            time.sleep(3)

if __name__ == "__main__":
    main()
