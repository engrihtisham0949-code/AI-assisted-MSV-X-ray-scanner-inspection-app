import base64, io, os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from groq import Groq

st.set_page_config(page_title="MSV AI Scanner Inspector", page_icon="🔎", layout="wide")
st.title("🔎 MSV Scanner AI Inspector")
st.caption("EMPTY • NORMAL • ABNORMAL | Density + Continuous Pattern Analysis")

ROWS, COLS = 12, 12
VISION_MODEL = "openai/gpt-oss-120b"

def groq_client():
    try: key = st.secrets.get("GROQ_API_KEY")
    except Exception: key = None
    key = key or os.getenv("GROQ_API_KEY")
    return Groq(api_key=key) if key else None

def zscore(a):
    med = np.median(a); mad = np.median(np.abs(a-med))
    if mad < 1e-6:
        sd = np.std(a)
        return np.zeros_like(a) if sd < 1e-6 else (a-med)/(sd+1e-6)
    return .6745*(a-med)/(mad+1e-6)

def neighbors(a):
    out = np.zeros_like(a, dtype=np.float32)
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            n=[]
            if r: n.append(a[r-1,c])
            if r<a.shape[0]-1: n.append(a[r+1,c])
            if c: n.append(a[r,c-1])
            if c<a.shape[1]-1: n.append(a[r,c+1])
            out[r,c]=abs(a[r,c]-np.mean(n))
    return out

def roi_from_image(gray):
    h,w=gray.shape
    box=(int(.03*w),int(.30*h),int(.97*w),int(.76*h))
    x1,y1,x2,y2=box
    return gray[y1:y2,x1:x2],box

def empty_test(roi, threshold):
    g=cv2.GaussianBlur(roi,(5,5),0)
    gx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3)
    gy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)
    mag=cv2.magnitude(gx,gy)
    my=max(1,int(.08*roi.shape[0])); mx=max(1,int(.03*roi.shape[1]))
    inner=mag[my:-my,mx:-mx]
    score=float(np.mean(inner if inner.size else mag))
    return score<threshold,score

def analyze(roi, dthr, pthr, cthr):
    h,w=roi.shape
    den=np.zeros((ROWS,COLS),np.float32)
    pat=np.zeros_like(den); tex=np.zeros_like(den)
    gx=cv2.Sobel(roi,cv2.CV_32F,1,0,ksize=3)
    gy=cv2.Sobel(roi,cv2.CV_32F,0,1,ksize=3)
    grad=cv2.magnitude(gx,gy)
    for r in range(ROWS):
        for c in range(COLS):
            y1,y2=int(r*h/ROWS),int((r+1)*h/ROWS)
            x1,x2=int(c*w/COLS),int((c+1)*w/COLS)
            t=roi[y1:y2,x1:x2]; g=grad[y1:y2,x1:x2]
            den[r,c]=np.median(t); pat[r,c]=np.mean(g); tex[r,c]=np.std(t)

    # STRICT RULE: any meaningful density difference OR pattern/continuity
    # break makes the image ABNORMAL.
    da=np.abs(zscore(den))>=dthr
    pa=(np.abs(zscore(neighbors(pat)))>=pthr) | (np.abs(zscore(neighbors(tex)))>=pthr)
    ca=np.abs(zscore(neighbors(den)))>=cthr
    abnormal=da|pa|ca
    return abnormal,int(da.sum()),int(pa.sum()),int(ca.sum())

def make_result(bgr, abnormal, box):
    x1,y1,x2,y2=box
    tile=(abnormal.astype(np.uint8)*255)
    roi_mask=cv2.resize(tile,(x2-x1,y2-y1),interpolation=cv2.INTER_NEAREST)
    roi_mask=cv2.morphologyEx(roi_mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    full=np.zeros(bgr.shape[:2],np.uint8); full[y1:y2,x1:x2]=roi_mask
    out=bgr.copy(); red=out.copy(); red[full>0]=(0,0,255)
    out=cv2.addWeighted(out,.65,red,.35,0)
    boxes=[]
    contours,_=cv2.findContours(full,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,w,h=cv2.boundingRect(cnt)
        if w*h>500:
            cv2.rectangle(out,(x,y),(x+w,y+h),(0,0,255),3)
            cv2.putText(out,"ABNORMAL",(x,max(25,y-8)),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2)
            boxes.append({"x":int(x),"y":int(y),"width":int(w),"height":int(h)})
    return out,boxes

def groq_review(image, local_status):
    client=groq_client()
    if not client: return "Groq API key not configured. Local inspection result is shown."
    buf=io.BytesIO(); image.save(buf,format="JPEG",quality=90)
    data=base64.b64encode(buf.getvalue()).decode()
    prompt=f"""Review this MSV/X-ray scanner image. Local result: {local_status}.
Rules: EMPTY if no meaningful cargo/item is inside the container. NORMAL only if cargo follows a continuous, consistent or repeated pattern. ABNORMAL if any meaningful portion has different density, or the continuous/repeated pattern is broken, discontinuous or inconsistent. STRICT: cargo that does not follow the expected consistent pattern is ABNORMAL. Give Status, Reason, and Suspicious area. This is a prototype; do not claim a proven security threat."""
    try:
        r=client.chat.completions.create(model=VISION_MODEL,messages=[{"role":"user","content":[{"type":"text","text":prompt},{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+data}}]}],temperature=.1,max_completion_tokens=350)
        return r.choices[0].message.content
    except Exception as e: return f"Groq second opinion unavailable: {e}"

with st.sidebar:
    st.header("Detection Settings")
    empty_thr=st.slider("EMPTY sensitivity",1.0,40.0,8.0,.5)
    density_thr=st.slider("Density anomaly threshold",1.0,6.0,2.8,.1)
    pattern_thr=st.slider("Pattern break threshold",1.0,6.0,2.8,.1)
    continuity_thr=st.slider("Continuity break threshold",1.0,6.0,2.8,.1)
    st.info("STRICT RULE: No cargo = EMPTY. Continuous/consistent cargo = NORMAL. Different density OR broken/inconsistent pattern = ABNORMAL.")

file=st.file_uploader("Upload MSV Scanner Image",type=["jpg","jpeg","png","bmp","webp"])
if not file:
    st.markdown("### Classification Logic\n- **EMPTY:** no meaningful item inside the container.\n- **NORMAL:** cargo has a continuous and consistent pattern.\n- **ABNORMAL:** different density or broken/inconsistent pattern. Abnormal regions are marked in red.")
    st.stop()

image=Image.open(file).convert("RGB")
bgr=cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
gray=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8)).apply(gray)
roi,box=roi_from_image(gray)
empty,score=empty_test(roi,empty_thr)

dc=pc=cc=0; boxes=[]
if empty:
    status="EMPTY"; result=bgr.copy()
else:
    abnormal,dc,pc,cc=analyze(roi,density_thr,pattern_thr,continuity_thr)
    status="ABNORMAL" if abnormal.any() else "NORMAL"
    result,boxes=make_result(bgr,abnormal,box)

a,b=st.columns(2)
with a:
    st.subheader("Original MSV Image"); st.image(image,use_container_width=True)
with b:
    st.subheader("Inspection Result"); st.image(cv2.cvtColor(result,cv2.COLOR_BGR2RGB),use_container_width=True)

if status=="EMPTY": st.success("🟢 EMPTY — No meaningful cargo/item detected inside the container.")
elif status=="NORMAL": st.success("🟢 NORMAL — Cargo follows a continuous and consistent pattern.")
else: st.error("🔴 ABNORMAL — Density difference or broken/inconsistent pattern detected. Red areas are suspicious.")

m1,m2,m3,m4=st.columns(4)
m1.metric("Final Status",status); m2.metric("Content Score",f"{score:.2f}")
m3.metric("Density Anomalies",dc); m4.metric("Pattern/Continuity",pc+cc)
if status=="ABNORMAL" and boxes:
    st.subheader("Marked Abnormal Regions"); st.json(boxes)

st.subheader("Groq AI Second Opinion")
with st.spinner("Groq is reviewing the image..."):
    st.write(groq_review(image,status))

st.warning("Prototype only. For operational use, calibrate the container ROI and thresholds using real labelled EMPTY, NORMAL and ABNORMAL MSV scanner images.")
