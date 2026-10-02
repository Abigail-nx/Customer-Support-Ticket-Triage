import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, re
from pathlib import Path
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import plotly.express as px

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
MODEL_DIR = APP_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
TICKETS_FILE = DATA_DIR / "tickets.csv"
TRAIN_FILE = DATA_DIR / "training_data.csv"
CAT_MODEL = MODEL_DIR / "category_model.joblib"
URG_MODEL = MODEL_DIR / "urgency_model.joblib"
METRICS_FILE = MODEL_DIR / "metrics.json"

CATEGORIES = ["Billing", "Technical", "Account", "Product", "Other"]
URGENCIES = ["Low", "Medium", "High", "Critical"]
QUEUES = {"Billing":"Billing Support", "Technical":"Technical Support",
          "Account":"Account Support", "Product":"Product Support", "Other":"General Support"}

st.set_page_config(page_title="AI Ticket Triage", page_icon="🎫", layout="wide")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def make_training_data():
    # Starter examples let the app run immediately. Replace with reviewed historical tickets for real use.
    examples = {
      "Billing": [
        "I was charged twice for my monthly subscription", "Please refund the payment taken from my card",
        "My invoice has the wrong amount", "The coupon did not apply and I paid full price",
        "I cannot download my receipt", "My payment failed but money was deducted",
        "There is an unexpected charge on my account", "Please explain this billing statement",
        "I need to update the card used for payment", "My subscription renewal was billed incorrectly",
        "I was charged after cancelling my plan", "The tax on my invoice looks incorrect"
      ],
      "Technical": [
        "The application crashes whenever I try to log in", "The website keeps showing a server error",
        "I cannot upload my document because the page freezes", "The mobile app is extremely slow today",
        "I get an error message when saving changes", "Notifications have stopped working",
        "The dashboard is not loading on my computer", "The system keeps disconnecting",
        "I am unable to export my report", "The latest update broke the search feature",
        "The page displays a blank screen", "I cannot connect the software to the printer"
      ],
      "Account": [
        "I forgot my password and cannot reset it", "Please change the email address on my account",
        "My account has been locked", "I need help enabling two factor authentication",
        "I want to delete my account and personal data", "I cannot verify my phone number",
        "Someone changed my account details without permission", "Please help me update my profile",
        "I am unable to access my account", "My verification code never arrives",
        "I need to change the account owner", "How can I recover my username"
      ],
      "Product": [
        "Does this plan include advanced reporting", "The product feature does not meet the listed specification",
        "I need information about the premium package", "How do I use the new collaboration feature",
        "The item I received is damaged", "Can you explain the product warranty",
        "I would like to request a feature for the next release", "The product description is unclear",
        "Is this device compatible with my operating system", "I need help choosing the right plan",
        "The product is missing an advertised accessory", "Please share the setup guide"
      ],
      "Other": [
        "Where can I find your office address", "I would like to share feedback about the service",
        "Can you tell me your customer support hours", "I have a general question",
        "Please connect me with the right department", "I want to know about your company",
        "Thank you for the excellent service", "I have a question about your policies",
        "Can someone call me back", "I would like to make a suggestion",
        "Where can I find the user manual", "I need information not covered in the help center"
      ]
    }
    urgency_phrases = {
      "Low": ["when convenient", "no rush", "for future reference", "whenever possible"],
      "Medium": ["please help me", "I need assistance", "could you check this", "please advise"],
      "High": ["this is urgent", "I need this resolved today", "blocking my work", "please respond soon"],
      "Critical": ["critical outage", "all users are affected", "business is stopped", "security breach"]
    }
    rows=[]
    for category, texts in examples.items():
        for i, text in enumerate(texts):
            for urgency, phrases in urgency_phrases.items():
                # Keep every category represented across all urgency labels.
                rows.append({"text": text + ". " + phrases[(i + len(category)) % len(phrases)] + ".",
                             "category": category, "urgency": urgency})
    return pd.DataFrame(rows)

def train_models(df):
    df = df.dropna(subset=["text","category","urgency"]).copy()
    df["text"] = df["text"].astype(str).map(clean_text)
    X_train, X_test, ycat_train, ycat_test, yurg_train, yurg_test = train_test_split(
        df["text"], df["category"], df["urgency"], test_size=.25, random_state=42, stratify=df["category"])
    cat = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=12000, sublinear_tf=True)),
                    ("clf", LogisticRegression(max_iter=1500, class_weight="balanced"))])
    urg = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=12000, sublinear_tf=True)),
                    ("clf", LogisticRegression(max_iter=1500, class_weight="balanced"))])
    cat.fit(X_train, ycat_train); urg.fit(X_train, yurg_train)
    cat_pred=cat.predict(X_test); urg_pred=urg.predict(X_test)
    metrics = {
      "category_accuracy": float(accuracy_score(ycat_test, cat_pred)),
      "category_f1_weighted": float(f1_score(ycat_test, cat_pred, average="weighted", zero_division=0)),
      "urgency_accuracy": float(accuracy_score(yurg_test, urg_pred)),
      "urgency_f1_weighted": float(f1_score(yurg_test, urg_pred, average="weighted", zero_division=0)),
      "category_report": classification_report(ycat_test, cat_pred, output_dict=True, zero_division=0),
      "urgency_report": classification_report(yurg_test, urg_pred, output_dict=True, zero_division=0),
      "category_labels": sorted(df.category.unique().tolist()),
      "urgency_labels": sorted(df.urgency.unique().tolist()),
      "category_cm": confusion_matrix(ycat_test, cat.labels_ if False else cat_pred, labels=sorted(df.category.unique())).tolist(),
      "category_cm_labels": sorted(df.category.unique().tolist()),
      "urgency_cm": confusion_matrix(yurg_test, urg_pred, labels=sorted(df.urgency.unique())).tolist(),
      "urgency_cm_labels": sorted(df.urgency.unique().tolist()),
      "test_size": int(len(X_test)), "training_size": int(len(X_train))
    }
    joblib.dump(cat, CAT_MODEL); joblib.dump(urg, URG_MODEL)
    METRICS_FILE.write_text(json.dumps(metrics, indent=2))
    return metrics

def load_or_train():
    if not TRAIN_FILE.exists():
        make_training_data().to_csv(TRAIN_FILE, index=False)
    df = pd.read_csv(TRAIN_FILE)
    if not CAT_MODEL.exists() or not URG_MODEL.exists() or not METRICS_FILE.exists():
        train_models(df)
    return joblib.load(CAT_MODEL), joblib.load(URG_MODEL), json.loads(METRICS_FILE.read_text())

def load_tickets():
    if TICKETS_FILE.exists():
        return pd.read_csv(TICKETS_FILE).fillna("")
    cols=["ticket_id","created_at","customer","text","category","urgency","queue","category_confidence","urgency_confidence","status","reviewed","actual_category","actual_urgency"]
    return pd.DataFrame(columns=cols)

def save_tickets(df):
    df.to_csv(TICKETS_FILE,index=False)

cat_model, urg_model, metrics = load_or_train()
tickets=load_tickets()

st.title("🎫 AI Customer Support Ticket Triage")
st.caption("Classify incoming customer requests, estimate urgency, and route tickets to the right queue.")
with st.sidebar:
    st.header("Navigation")
    page=st.radio("Open",["Submit Ticket","Ticket Queue","Analytics","Model & Data"])
    st.divider()
    st.caption("Demo models are trained on starter examples. Replace them with reviewed historical tickets before production use.")

if page=="Submit Ticket":
    st.subheader("Create and triage a ticket")
    with st.form("new_ticket", clear_on_submit=True):
        customer=st.text_input("Customer name (optional)")
        ticket_text=st.text_area("Customer message", placeholder="Example: I was charged twice and need a refund today...", height=150)
        submitted=st.form_submit_button("Predict and route", type="primary")
    if submitted:
        if not ticket_text.strip():
            st.error("Please enter a customer message.")
        else:
            X=[clean_text(ticket_text)]
            cp=cat_model.predict_proba(X)[0]; up=urg_model.predict_proba(X)[0]
            ci=int(np.argmax(cp)); ui=int(np.argmax(up))
            category=cat_model.classes_[ci]; urgency=urg_model.classes_[ui]
            cc=float(cp[ci]); uc=float(up[ui])
            tid=f"TKT-{datetime.now().strftime('%y%m%d%H%M%S')}-{len(tickets)+1:03d}"
            status="Needs review" if min(cc,uc)<0.55 else "Open"
            row={"ticket_id":tid,"created_at":datetime.now().isoformat(timespec="seconds"),"customer":customer or "Guest",
                 "text":ticket_text,"category":category,"urgency":urgency,"queue":QUEUES.get(category,"General Support"),
                 "category_confidence":round(cc,4),"urgency_confidence":round(uc,4),"status":status,
                 "reviewed":"No","actual_category":"","actual_urgency":""}
            tickets=pd.concat([tickets,pd.DataFrame([row])],ignore_index=True); save_tickets(tickets)
            st.success(f"Ticket {tid} created.")
            a,b,c=st.columns(3)
            a.metric("Predicted category",category); b.metric("Urgency",urgency); c.metric("Assigned queue",row["queue"])
            st.write(f"Category confidence: **{cc:.1%}** · Urgency confidence: **{uc:.1%}**")
            if status=="Needs review":
                st.warning("Low-confidence prediction: this ticket has been flagged for human review.")
            st.info("Predictions are estimates. A human agent should verify high-impact or uncertain tickets.")

elif page=="Ticket Queue":
    st.subheader("Incoming ticket queue")
    if tickets.empty:
        st.info("No tickets yet. Submit a sample ticket to get started.")
    else:
        f1,f2,f3=st.columns(3)
        categories=["All"]+sorted(tickets.category.astype(str).unique().tolist())
        urgencies=["All"]+sorted(tickets.urgency.astype(str).unique().tolist())
        statuses=["All"]+sorted(tickets.status.astype(str).unique().tolist())
        cf=f1.selectbox("Category",categories); uf=f2.selectbox("Urgency",urgencies); sf=f3.selectbox("Status",statuses)
        view=tickets.copy()
        if cf!="All": view=view[view.category==cf]
        if uf!="All": view=view[view.urgency==uf]
        if sf!="All": view=view[view.status==sf]
        st.caption(f"{len(view)} ticket(s)")
        for idx,row in view.iloc[::-1].iterrows():
            with st.expander(f"{row.ticket_id} · {row.category} · {row.urgency} · {row.status}"):
                st.write(row.text)
                st.caption(f"Customer: {row.customer} | Queue: {row.queue} | Created: {row.created_at}")
                st.write(f"Model confidence — category: {float(row.category_confidence):.1%}, urgency: {float(row.urgency_confidence):.1%}")
                with st.form(f"review_{row.ticket_id}"):
                    c1,c2=st.columns(2)
                    actual_cat=c1.selectbox("Correct category",CATEGORIES,index=CATEGORIES.index(row.category) if row.category in CATEGORIES else 0,key=f"cat_{row.ticket_id}")
                    actual_urg=c2.selectbox("Correct urgency",URGENCIES,index=URGENCIES.index(row.urgency) if row.urgency in URGENCIES else 0,key=f"urg_{row.ticket_id}")
                    new_status=st.selectbox("Status",["Open","In progress","Resolved","Needs review"],index=["Open","In progress","Resolved","Needs review"].index(row.status) if row.status in ["Open","In progress","Resolved","Needs review"] else 0,key=f"status_{row.ticket_id}")
                    if st.form_submit_button("Save human review"):
                        tickets.loc[idx,"actual_category"]=actual_cat
                        tickets.loc[idx,"actual_urgency"]=actual_urg
                        tickets.loc[idx,"reviewed"]="Yes"
                        tickets.loc[idx,"category"]=actual_cat
                        tickets.loc[idx,"urgency"]=actual_urg
                        tickets.loc[idx,"queue"]=QUEUES[actual_cat]
                        tickets.loc[idx,"status"]=new_status
                        save_tickets(tickets); st.rerun()

elif page=="Analytics":
    st.subheader("Ticket analytics")
    if tickets.empty:
        st.info("Analytics will appear after tickets have been submitted.")
    else:
        a,b,c,d=st.columns(4)
        a.metric("Total tickets",len(tickets))
        b.metric("Open / active",int(tickets.status.isin(["Open","In progress","Needs review"]).sum()))
        c.metric("Needs review",int((tickets.status=="Needs review").sum()))
        d.metric("Human reviewed",int((tickets.reviewed=="Yes").sum()))
        left,right=st.columns(2)
        with left:
            st.plotly_chart(px.histogram(tickets,x="category",color="category",title="Tickets by category"),use_container_width=True)
        with right:
            st.plotly_chart(px.histogram(tickets,x="urgency",color="urgency",category_orders={"urgency":URGENCIES},title="Tickets by urgency"),use_container_width=True)
        st.plotly_chart(px.histogram(tickets,x="queue",color="status",title="Queue workload"),use_container_width=True)
        st.download_button("Download ticket data (CSV)",tickets.to_csv(index=False).encode("utf-8"),"tickets.csv","text/csv")

else:
    st.subheader("Model performance & training data")
    a,b,c,d=st.columns(4)
    a.metric("Category accuracy",f"{metrics['category_accuracy']:.1%}")
    b.metric("Category weighted F1",f"{metrics['category_f1_weighted']:.1%}")
    c.metric("Urgency accuracy",f"{metrics['urgency_accuracy']:.1%}")
    d.metric("Urgency weighted F1",f"{metrics['urgency_f1_weighted']:.1%}")
    st.warning("These metrics are from a small built-in demonstration dataset and are not a reliable estimate of real-world performance.")
    cm1,cm2=st.columns(2)
    with cm1:
        st.write("Category confusion matrix")
        st.dataframe(pd.DataFrame(metrics["category_cm"],index=metrics["category_cm_labels"],columns=metrics["category_cm_labels"]),use_container_width=True)
    with cm2:
        st.write("Urgency confusion matrix")
        st.dataframe(pd.DataFrame(metrics["urgency_cm"],index=metrics["urgency_cm_labels"],columns=metrics["urgency_cm_labels"]),use_container_width=True)
    st.markdown("#### Train with historical tickets")
    st.write("Upload a CSV with columns `text`, `category`, and `urgency`. Categories: Billing, Technical, Account, Product, Other. Urgency: Low, Medium, High, Critical.")
    upload=st.file_uploader("Historical ticket CSV",type=["csv"])
    if upload is not None:
        try:
            new_df=pd.read_csv(upload)
            required={"text","category","urgency"}
            if not required.issubset(new_df.columns): st.error("CSV must contain text, category, and urgency columns.")
            elif not set(new_df.category.dropna()).issubset(CATEGORIES): st.error("Unknown category found. Use the listed category labels.")
            elif not set(new_df.urgency.dropna()).issubset(URGENCIES): st.error("Unknown urgency found. Use the listed urgency labels.")
            elif len(new_df)<30: st.error("Please provide at least 30 labeled rows for a basic train/test split.")
            else:
                st.success(f"Loaded {len(new_df)} rows.")
                if st.button("Train / retrain models",type="primary"):
                    new_df.to_csv(TRAIN_FILE,index=False)
                    train_models(new_df)
                    st.success("Models retrained. Reloading the dashboard...")
                    st.rerun()
        except Exception as e: st.error(f"Could not read CSV: {e}")
    if st.button("Restore demo training data"):
        make_training_data().to_csv(TRAIN_FILE,index=False)
        for p in [CAT_MODEL,URG_MODEL,METRICS_FILE]:
            if p.exists(): p.unlink()
        st.success("Demo data restored. Reloading...")
        st.rerun()

st.divider()
st.caption("Prototype for learning and demonstration. Review predictions before taking customer-impacting actions; do not upload sensitive personal data into an unapproved environment.")
