import streamlit as st
import pandas as pd
from collections import deque

# --- State Machines Logic ---

class MajorityVoting:
    """מחלקה שמנהלת חלון זז של 5 פריימים ומחזירה את ערך הרוב (3/5)"""
    def __init__(self, window_size=5):
        self.q = deque(maxlen=window_size)
        
    def push(self, val):
        self.q.append(val)
        
    def get_majority(self):
        if len(self.q) < 3:
            return None
        counts = {}
        for v in self.q:
            counts[v] = counts.get(v, 0) + 1
        for v, count in counts.items():
            if count >= 3:
                return v
        return None

class RepStateMachine:
    """מכונת מצבים לספירת חזרות לפי התרשים"""
    def __init__(self):
        self.state = 'idle'
        self.vote = MajorityVoting(5)
        self.rep_count = 0
        
    def process(self, phase_val):
        # הופכים לאותיות קטנות כדי למנוע בעיות של אותיות גדולות/קטנות
        self.vote.push(str(phase_val).strip().lower())
        maj = self.vote.get_majority()
        output = None
        
        if maj is None:
            return output, self.state
            
        # מעברי המצבים (משתמשים ב-in כדי לתפוס גם חלקי מילים)
        if self.state == 'idle':
            if 'high' in maj: 
                self.state = '1High'
                
        elif self.state == '1High':
            if 'medium' in maj or 'mid' in maj: 
                self.state = '1Mid'
            elif 'low' in maj: 
                output = -1 # Error
                
        elif self.state == '1Mid':
            if 'low' in maj: 
                self.state = 'Low'
            elif 'high' in maj:
                self.state = '1High'
                output = 2 # Half way up repetition
                
        elif self.state == 'Low':
            if 'medium' in maj or 'mid' in maj: 
                self.state = '2Mid'
            elif 'high' in maj:
                self.state = '1High'
                output = -1 # Error
                
        elif self.state == '2Mid':
            if 'high' in maj:
                self.state = '1High'
                output = 1 # Full Rep Completed!
                self.rep_count += 1
            elif 'low' in maj:
                self.state = 'Low'
                output = 3 # Half way down repetition
                
        return output, self.state

class HipStateMachine:
    """מכונת מצבים לזיהוי מנח אגן"""
    def __init__(self):
        self.vote = MajorityVoting(5)
        
    def process(self, hip_val):
        # הופכים לאותיות קטנות חסינות תקלות
        self.vote.push(str(hip_val).strip().lower())
        maj = self.vote.get_majority()
        
        if maj is None: 
            return -1
        
        # חיפוש חלקי של המילה (תופס 'Low', 'too low', 'TOO LOW' וכו')
        if 'high' in maj: return 1       # אגן גבוה מדי
        elif 'low' in maj: return 2      # אגן נמוך מדי
        elif 'good' in maj: return 0     # אגן תקין
        else: return -1                  # שגיאה / ללא רוב

# --- Streamlit UI & Data Processing ---

st.set_page_config(page_title="Offline CSV Validator", layout="wide")
st.title("🏋️‍♂️ A-Eye Trainer: Offline CSV Validator")
st.markdown("Upload your marked CSV file to run the sequence through the state machines.")

uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # ניקוי שמות עמודות כדי למנוע שגיאות רווחים
    df.columns = df.columns.str.strip().str.lower()
    
    # מציאת העמודות הרלוונטיות באופן דינמי
    hip_col = [c for c in df.columns if 'hip' in c][0]
    phase_col = [c for c in df.columns if 'pushup_ph' in c or 'phase' in c][0]
    
    # הוספת חיפוש לעמודת הפריים החוקי
    valid_col = [c for c in df.columns if 'valid' in c][0]
    
    st.write(f"**Loaded {len(df)} frames.**")
    
    if st.button("Run State Machines", type="primary"):
        rep_sm = RepStateMachine()
        hip_sm = HipStateMachine()
        
        bad_hip_frames = []
        events_log = []
        invalid_frames_count = 0
        
        # לולאת ריצה על כל פריים ב-CSV
        for index, row in df.iterrows():
            frame_num = index + 1 # מספרי פריימים מתחילים ב-1
            
            # 1. בדיקה האם הפריים חוקי
            is_valid = str(row[valid_col]).strip().upper()
            if is_valid == 'FALSE' or is_valid == '0':
                invalid_frames_count += 1
                continue # דילוג על שאר הלוגיקה עבור פריים זה
                
            # 2. קריאת הנתונים רק אם הפריים חוקי
            current_hip = str(row[hip_col]).strip()
            current_phase = str(row[phase_col]).strip()
            
            # 3. הרצת הלוגיקה
            rep_output, current_rep_state = rep_sm.process(current_phase)
            hip_output = hip_sm.process(current_hip)
            
            # תיעוד חזרות ואירועים חריגים ממונה החזרות
            if rep_output == 1:
                events_log.append(f"✅ Frame {frame_num}: Full repetition completed (Total: {rep_sm.rep_count})")
            elif rep_output == 2:
                events_log.append(f"⚠️ Frame {frame_num}: Aborted Rep (Half way up)")
            elif rep_output == 3:
                events_log.append(f"⚠️ Frame {frame_num}: Aborted Rep (Half way down)")
            elif rep_output == -1:
                events_log.append(f"❌ Frame {frame_num}: Sequence Error (Jumped states)")
                
            # תיעוד מנח אגן לא תקין ושמירת מספר הפריים
            if hip_output == 1:
                bad_hip_frames.append({'frame': frame_num, 'issue': 'Too High'})
            elif hip_output == 2:
                bad_hip_frames.append({'frame': frame_num, 'issue': 'Too Low'})

        # --- Display Results ---
        st.divider()
        
        # הצגת כמות הפריימים הלא חוקיים שסוננו
        if invalid_frames_count > 0:
            st.info(f"⏭️ Skipped {invalid_frames_count} invalid frames during processing.")
            
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("📊 Repetition Analysis")
            st.metric("Total Valid Reps", rep_sm.rep_count)
            if events_log:
                st.write("**Sequence Events:**")
                for event in events_log:
                    st.text(event)
            else:
                st.write("No special events or full reps completed.")
                
        with col2:
            st.header("🛑 Hip Form Alerts")
            if not bad_hip_frames:
                st.success("Perfect Hip Form! No bad frames detected.")
            else:
                st.warning(f"Detected {len(bad_hip_frames)} valid frames with incorrect hip form.")
                bad_df = pd.DataFrame(bad_hip_frames)
                st.dataframe(bad_df, use_container_width=True)