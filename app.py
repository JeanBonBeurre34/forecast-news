"""
app.py — Global Trend Forecasting Pipeline (Readable Dashboard Edition)
------------------------------------------------------------------------
• Fetches up to 5 years of GDELT tone-weighted data
• Aggregates weekly counts for Prophet forecasting
• Uses >20 international & tech RSS feeds
• Tracks geopolitics, energy, AI, cyber & emerging innovation
• Produces readable high-res PNGs: top 8, dimmed full, and category subplots
"""

import os, re, traceback
from datetime import datetime
import requests, feedparser, pandas as pd
from newspaper import Article
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.feature_extraction.text import CountVectorizer

# ---------- Matplotlib style ----------
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 16,
    "legend.fontsize": 10
})

# ---------- Config ----------
FEEDS = [
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "https://rss.cnn.com/rss/edition_world.rss",
    "https://www.reuters.com/rssFeed/worldNews",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://www.theguardian.com/world/rss",
    "https://www.theguardian.com/environment/rss",
    "https://www.france24.com/en/rss",
    "https://rss.dw.com/rdf/rss-en-all",
    "https://www.npr.org/rss/rss.php?id=1004",
    "https://www.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://www.euronews.com/rss?level=theme&name=news",
    "https://www.bloomberg.com/feed/podcast/global-news.rss",
    "https://www.wired.com/feed/rss",
    "https://feeds.arstechnica.com/arstechnica/index/",
    "https://techcrunch.com/feed/",
    "https://www.engadget.com/rss.xml",
    "https://www.popularmechanics.com/rss/all.xml/",
    "https://www.scientificamerican.com/feed/",
    "https://www.nature.com/subjects/technology/rss",
    "https://www.mittechreview.com/feed/",
    "https://cleantechnica.com/feed/",
    "https://www.energy.gov/rss/energy-news.xml",
    "https://electrek.co/feed/",
    "https://www.space.com/feeds/all",
    "https://www.inverse.com/rss"
]

BASE_TOPICS = [
    "Ukraine war","AI","energy","cyberattack","climate","election",
    "inflation","migration","nuclear","conflict","sanctions",
    "fusion","solar panel","electric vehicle","cybernetic implant"
]

DISCOVER_LIMIT = 5
ARTICLE_LIMIT = 40
SAVE_DIR = "data"
os.makedirs(SAVE_DIR, exist_ok=True)
HISTORY_FILE = os.path.join(SAVE_DIR, "topic_history.csv")
ACTIONS_LOG = os.path.join(SAVE_DIR, "actions.log")
ERRORS_LOG  = os.path.join(SAVE_DIR, "errors.log")
RUN_CHECK   = os.path.join(SAVE_DIR, "run_check.txt")

# ---------- Logging ----------
def log_action(msg):
    ts = datetime.now().isoformat()
    print(f"{ts}  ACTION: {msg}")
    with open(ACTIONS_LOG,"a") as f: f.write(f"{ts}  ACTION: {msg}\n")

def log_error(e):
    ts = datetime.now().isoformat()
    trace = traceback.format_exc()
    print(f"{ts}  ERROR: {e}")
    with open(ERRORS_LOG,"a") as f: f.write(f"{ts}  ERROR: {repr(e)}\n{trace}\n\n")

def safe_save_df(df,path,index=True):
    try:
        df.to_csv(path,index=index)
        log_action(f"Saved CSV -> {path}")
    except Exception as e: log_error(e)

def safe_save_plot(fig,path):
    try:
        fig.savefig(path,dpi=200,bbox_inches="tight")
        plt.close(fig)
        log_action(f"Saved high-res plot -> {path}")
    except Exception as e: log_error(e)

def write_run_check():
    with open(RUN_CHECK,"w") as f:
        f.write(f"run_check at {datetime.now().isoformat()}\n")
    log_action(f"Wrote run check -> {RUN_CHECK}")

# ---------- GDELT ----------
def fetch_historical_news(topic):
    base_url="https://api.gdeltproject.org/api/v2/doc/doc"
    params={"query":topic,"mode":"timelinetone","timespan":"5y","format":"json"}
    try:
        r=requests.get(base_url,params=params,timeout=25)
        if r.status_code!=200 or not r.text.strip():
            log_action(f"[!] GDELT HTTP {r.status_code} for '{topic}'");return pd.DataFrame()
        try: payload=r.json()
        except Exception as e:
            log_action(f"[!] JSON decode failed for '{topic}': {e}");return pd.DataFrame()
        timeline=payload.get("timeline") or payload.get("TIMELINE") or []
        if not timeline:
            log_action(f"[!] No usable GDELT timeline for '{topic}'");return pd.DataFrame()
        df=pd.DataFrame(timeline)
        date_col=next((c for c in ["date","datetime","time"] if c in df.columns),None)
        if not date_col or "value" not in df.columns:
            log_action(f"[!] Invalid GDELT data for '{topic}'");return pd.DataFrame()
        df["date"]=pd.to_datetime(df[date_col],errors="coerce").dt.date
        df=df.rename(columns={"value":topic})[["date",topic]]
        df=df.dropna(subset=["date"]).set_index("date")
        df[topic]=pd.to_numeric(df[topic],errors="coerce").fillna(0)
        log_action(f"[+] GDELT OK for '{topic}' ({len(df)} rows)")
        return df
    except Exception as e:
        log_error(e);log_action(f"GDELT fetch failed for '{topic}'");return pd.DataFrame()

# ---------- RSS ----------
def fetch_news(feeds,limit=ARTICLE_LIMIT):
    rows=[]
    for url in feeds:
        log_action(f"Loading RSS feed: {url}")
        try: feed=feedparser.parse(url)
        except Exception as e: log_error(e);continue
        for e in feed.entries[:limit]:
            link=getattr(e,"link","")
            try:
                art=Article(link); art.download(); art.parse()
                rows.append({"title":art.title,"text":art.text,
                             "date":art.publish_date or getattr(e,"published",None),
                             "source":url})
            except Exception as ex: log_action(f"Skipping article: {link} ({ex})")
    df=pd.DataFrame(rows)
    if df.empty: log_action("No RSS articles collected."); return df
    df["date"]=pd.to_datetime(df["date"],errors="coerce").dt.date
    df=df.dropna(subset=["date"])
    log_action(f"Collected {len(df)} live articles."); return df

# ---------- Keywords ----------
def discover_keywords(df,n=DISCOVER_LIMIT):
    texts=df["title"].fillna("")+" "+df["text"].fillna("")
    clean=texts.str.replace(r"[^A-Za-z\s]"," ",regex=True).str.lower()
    vectorizer=CountVectorizer(stop_words="english",ngram_range=(2,2),min_df=2,max_features=2000)
    try:
        X=vectorizer.fit_transform(clean); freqs=X.sum(axis=0).A1
        vocab=vectorizer.get_feature_names_out()
        top=(pd.DataFrame({"phrase":vocab,"count":freqs})
             .sort_values("count",ascending=False).head(n)["phrase"].tolist())
        log_action(f"Discovered keywords: {top}"); return top
    except Exception as e: log_error(e); return []

# ---------- Topic series ----------
def build_topic_series(df,topics):
    data=[]
    for topic in topics:
        pattern=re.escape(topic)
        try: counts=(df[df["text"].str.contains(pattern,case=False,na=False)]
                     .groupby("date").size())
        except Exception:
            counts=(df[df["text"].fillna("").str.contains(pattern,case=False)]
                    .groupby("date").size())
        data.append(counts.rename(topic))
    combined=pd.concat(data,axis=1).fillna(0)
    combined.index.name="date"
    log_action(f"Built topic series for {len(topics)} topics; shape={combined.shape}")
    return combined.astype(int)

# ---------- History ----------
def append_history(new_df,path=HISTORY_FILE):
    if isinstance(new_df.index,(pd.DatetimeIndex,pd.Index)) and new_df.index.name=="date":
        new_df=new_df.reset_index()
    if "date" not in new_df.columns:
        log_action("[!] append_history without 'date'; adding synthetic."); new_df["date"]=datetime.now().date()
    new_df["date"]=pd.to_datetime(new_df["date"]).dt.date
    if os.path.exists(path):
        try:
            old=pd.read_csv(path,parse_dates=["date"]); old["date"]=old["date"].dt.date
            merged=pd.concat([old,new_df],ignore_index=True)
            merged=merged.groupby("date").sum().sort_index()
            log_action(f"Merged existing {len(old)} + new {len(new_df)} rows into history.")
        except Exception as e: log_error(e); merged=new_df.copy()
    else: merged=new_df.copy()
    if "date" in merged.columns: merged=merged.set_index("date")
    merged.reset_index().to_csv(path,index=False)
    log_action(f"Appended history -> {path} (rows={len(merged)})")
    merged.index.name="date"; return merged

# ---------- Prophet ----------
def forecast_topic_trend(df,topic,weeks=260):
    if topic not in df.columns: return None,None
    try:
        data=df[[topic]].resample("W").sum().reset_index().rename(columns={"date":"ds",topic:"y"})
        data=data[data["y"]>0]
        if data.empty or len(data)<5:
            log_action(f"Skipping '{topic}' – insufficient weekly data ({len(data)})"); return None,None
        model=Prophet(weekly_seasonality=True,daily_seasonality=False)
        model.fit(data)
        future=model.make_future_dataframe(periods=weeks,freq="W")
        forecast=model.predict(future)
        forecast["trend"]=forecast["trend"].clip(lower=0)
        last_val=forecast.iloc[-weeks:]["trend"].mean()
        current_val=forecast.iloc[-(weeks+1)]["trend"]
        growth=(last_val-current_val)/max(current_val,1e-6)
        fig=model.plot(forecast,figsize=(16,9)); plt.tight_layout(pad=3)
        safe_save_plot(fig,os.path.join(SAVE_DIR,f"{topic.replace(' ','_')}_forecast.png"))
        return growth,forecast
    except Exception as e:
        log_error(e); log_action(f"Prophet failed for '{topic}'"); return None,None

# ---------- MAIN ----------
def main():
    write_run_check()
    try:
        log_action("Bootstrapping up to 5 years of GDELT data ...")
        hist_frames=[]
        for topic in BASE_TOPICS:
            df_hist=fetch_historical_news(topic)
            if not df_hist.empty: hist_frames.append(df_hist)
        hist_df=pd.concat(hist_frames,axis=1).fillna(0) if hist_frames else pd.DataFrame()
        if not hist_df.empty:
            hist_df.index.name="date"; safe_save_df(hist_df,os.path.join(SAVE_DIR,"hist_gdelt_raw.csv"))
        else: log_action("No GDELT data retrieved.")

        log_action("Fetching live RSS news ...")
        df_live=fetch_news(FEEDS)
        if df_live.empty and hist_df.empty: log_action("No data; exiting."); return
        safe_save_df(df_live,os.path.join(SAVE_DIR,"live_articles_raw.csv"),index=False)

        new_keywords=discover_keywords(df_live) if not df_live.empty else []
        topics=list(set(BASE_TOPICS+new_keywords))
        log_action(f"Tracking total topics: {len(topics)}")

        today_df=build_topic_series(df_live,topics) if not df_live.empty else pd.DataFrame()
        safe_save_df(today_df,os.path.join(SAVE_DIR,"today_topic_counts.csv"))

        if not hist_df.empty and not today_df.empty:
            all_topics=sorted(set(hist_df.columns.tolist()+today_df.columns.tolist()))
            hist_df=hist_df.reindex(columns=all_topics,fill_value=0)
            today_df=today_df.reindex(columns=all_topics,fill_value=0)
            combined_df=pd.concat([hist_df,today_df]).groupby(level=0).sum().fillna(0)
        elif not hist_df.empty: combined_df=hist_df.copy()
        else: combined_df=today_df.copy()

        topic_df=append_history(combined_df)
        safe_save_df(topic_df,os.path.join(SAVE_DIR,"topic_history.csv"))

        # ---------- Visualization ----------
        if not topic_df.empty:
            smooth_df=topic_df.rolling(window=4,min_periods=1).mean()  # 4-week smoothing

            # 1. Top-8 topics
            top_topics=smooth_df.sum().sort_values(ascending=False).head(8).index
            subset_df=smooth_df[top_topics]
            fig,ax=plt.subplots(figsize=(16,9))
            subset_df.plot(ax=ax,linewidth=2)
            ax.set_title("Top 8 Global Topics (Weekly Smoothed)")
            ax.legend(loc="upper left",fontsize=9,ncol=2,frameon=False)
            fig.tight_layout(pad=3)
            safe_save_plot(fig,os.path.join(SAVE_DIR,"topic_summary_top8.png"))

            # 2. Dimmed overview
            fig,ax=plt.subplots(figsize=(16,9))
            smooth_df.plot(ax=ax,linewidth=1.0,alpha=0.5)
            ax.set_title("All Topics Overview (Dimmed)")
            ax.legend([],[],frameon=False)
            fig.tight_layout(pad=3)
            safe_save_plot(fig,os.path.join(SAVE_DIR,"topic_summary_all.png"))

            # 3. Category subplots
            categories={
                "Energy_Tech":["energy","fusion","solar panel","electric vehicle"],
                "Cyber_AI":["AI","cyberattack","cybernetic implant"],
                "Geopolitics":["Ukraine war","sanctions","conflict","election"]
            }
            for name,topics_cat in categories.items():
                subset=smooth_df[smooth_df.columns.intersection(topics_cat)]
                if subset.empty: continue
                fig,ax=plt.subplots(figsize=(14,8))
                subset.plot(ax=ax,linewidth=2)
                ax.set_title(f"{name} Trends (Weekly Smoothed)")
                ax.legend(fontsize=9,ncol=2)
                fig.tight_layout(pad=3)
                safe_save_plot(fig,os.path.join(SAVE_DIR,f"topic_summary_{name}.png"))

        # ---------- Forecast ----------
        log_action("Forecasting trends (5-year weekly horizon) ...")
        growth_scores={}
        for topic in topic_df.columns.tolist():
            growth,_=forecast_topic_trend(topic_df,topic)
            if growth is not None: growth_scores[topic]=growth

        if not growth_scores:
            log_action("Not enough data for forecasts."); return
        sorted_topics=sorted(growth_scores.items(),key=lambda x:x[1],reverse=True)
        top_topic,top_growth=sorted_topics[0]
        log_action(f"🔥 Next trending topic: {top_topic} ({top_growth*100:+.1f}%)")

        safe_save_df(pd.Series(growth_scores).rename("growth").to_frame(),
                     os.path.join(SAVE_DIR,"forecast_growth.csv"))
        with open(os.path.join(SAVE_DIR,"latest.txt"),"w") as f:
            f.write(f"{datetime.now().isoformat()}\n")
            f.write(f"Next trending topic: {top_topic} ({top_growth*100:+.1f}%)\n")

        log_action(f"✅ Pipeline complete. Outputs in {os.path.abspath(SAVE_DIR)}")

    except Exception as exc:
        log_error(exc); log_action("Fatal error — see errors.log"); raise

if __name__=="__main__": main()
