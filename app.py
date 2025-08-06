import streamlit as st
import speech_recognition as sr
import pyttsx3
import spacy
from transformers import pipeline
import torch
import re
import tempfile
import os
import time
from typing import Dict, List, Tuple, Optional
import editdistance
import textstat
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# Configure Streamlit
st.set_page_config(
    page_title="Context-Aware Speech Correction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    .model-card {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
    }
    .grammar-valid {
        background: linear-gradient(45deg, #f0fdf4 0%, #dcfce7 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #16a34a;
        margin: 1rem 0;
        box-shadow: 0 3px 15px rgba(0,0,0,0.08);
    }
    .grammar-invalid {
        background: linear-gradient(45deg, #fef2f2 0%, #fee2e2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #dc2626;
        margin: 1rem 0;
        box-shadow: 0 3px 15px rgba(0,0,0,0.08);
    }
    .alternative-suggestion {
        background: linear-gradient(45deg, #fefbf3 0%, #fef3c7 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f59e0b;
        margin: 0.5rem 0;
    }
    .processing-step {
        background: #f8fafc;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        border-left: 3px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

class ContextAwareSpeechCorrector:
    def __init__(self):
        # Context-aware expansion dictionary with multiple possibilities
        self.context_expansions = {
            'wa': {
                'want': {'contexts': ['to', 'for', 'some'], 'probability': 0.7},
                'was': {'contexts': ['to', 'at', 'in'], 'probability': 0.2},
                'water': {'contexts': ['please', 'now', 'drink'], 'probability': 0.1}
            },
            'swi': {
                'swim': {'contexts': ['to', 'in', 'pool', 'water'], 'probability': 0.6},
                'switch': {'contexts': ['to', 'the', 'on', 'off'], 'probability': 0.3},
                'swing': {'contexts': ['on', 'the'], 'probability': 0.1}
            },
            'ea': {
                'eat': {'contexts': ['food', 'something', 'now'], 'probability': 0.6},
                'eating': {'contexts': ['food', 'lunch', 'dinner'], 'probability': 0.4}
            },
            'drin': {
                'drink': {'contexts': ['water', 'something', 'now'], 'probability': 0.5},
                'drinking': {'contexts': ['water', 'coffee', 'tea'], 'probability': 0.5}
            },
            'nee': {
                'need': {'contexts': ['to', 'help', 'water', 'food'], 'probability': 0.8},
                'knee': {'contexts': ['pain', 'hurt', 'injury'], 'probability': 0.2}
            },
            'hel': {
                'help': {'contexts': ['me', 'please', 'now'], 'probability': 0.9},
                'hello': {'contexts': [], 'probability': 0.1}
            },
            'foo': {
                'food': {'contexts': ['eat', 'hungry', 'want'], 'probability': 0.9},
                'foot': {'contexts': ['pain', 'hurt'], 'probability': 0.1}
            },
            'doc': {
                'doctor': {'contexts': ['see', 'visit', 'help'], 'probability': 0.8},
                'document': {'contexts': ['read', 'write'], 'probability': 0.2}
            },
            'med': {
                'medicine': {'contexts': ['take', 'need', 'doctor'], 'probability': 0.7},
                'medical': {'contexts': ['help', 'emergency'], 'probability': 0.3}
            },
            'pai': {
                'pain': {'contexts': ['have', 'feel', 'hurt'], 'probability': 0.8},
                'pay': {'contexts': ['money', 'bill'], 'probability': 0.2}
            }
        }
        
        # Grammar validation patterns
        self.invalid_patterns = [
            r'\b(I|you|he|she|we|they)\s+(water|food|medicine)\s+to\b',  # "I water to..."
            r'\bto\s+(water|food|eating|drinking)\b',  # "to water", "to food"
            r'\b(want|need)\s+(water|food)\s+(to|for)\s+\w+',  # "want water to swim"
            r'\b(am|is|are)\s+(to|for)\s+\w+ing\b',  # "am to swimming"
            r'\bwater\s+to\s+\w+',  # "water to swim"
            r'\bfood\s+to\s+\w+',   # "food to go"
        ]

    @st.cache_resource
    def load_models(_self):
        """Load required models"""
        models = {}
        
        try:
            st.info("🔄 Loading spaCy model...")
            models['spacy'] = spacy.load("en_core_web_sm")
            
            st.info("🔄 Loading grammar checker...")
            models['grammar_corrector'] = pipeline(
                "text2text-generation",
                model="vennify/t5-base-grammar-correction",
                device=-1
            )
            
            st.info("🔄 Loading context analyzer...")
            models['context_analyzer'] = pipeline(
                "fill-mask",
                model="bert-base-uncased",
                device=-1
            )
            
            st.success("✅ Models loaded successfully!")
            return models
            
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return {}

    def analyze_context(self, text: str, incomplete_word: str, position: int) -> str:
        """Analyze context to determine best word expansion"""
        words = text.split()
        
        # Get surrounding context
        prev_word = words[position-1].lower() if position > 0 else ""
        next_word = words[position+1].lower() if position < len(words)-1 else ""
        
        # Find best expansion based on context
        if incomplete_word in self.context_expansions:
            possibilities = self.context_expansions[incomplete_word]
            best_match = None
            highest_score = 0
            
            for expansion, info in possibilities.items():
                score = info['probability']
                
                # Boost score based on context matches
                for context_word in info['contexts']:
                    if context_word in [prev_word, next_word]:
                        score += 0.3
                    if context_word in text.lower():
                        score += 0.1
                
                if score > highest_score:
                    highest_score = score
                    best_match = expansion
            
            return best_match if best_match else incomplete_word
        
        return incomplete_word

    def context_aware_expansion(self, text: str) -> str:
        """Apply context-aware word expansion with better logic"""
        words = text.split()
        corrected_words = []
        
        for i, word in enumerate(words):
            # Check if word is potentially incomplete
            if len(word) <= 3 and word.isalpha() and word.lower() in self.context_expansions:
                
                # Get context
                prev_word = words[i-1].lower() if i > 0 else ""
                next_word = words[i+1].lower() if i < len(words)-1 else ""
                
                # Special handling for common problematic cases
                if word.lower() == "wa":
                    if next_word == "to" or (i < len(words)-2 and words[i+2].lower() in ["swim", "go", "come", "switch"]):
                        corrected_words.append("want")  # "I want to swim"
                    elif prev_word in ["i", "you", "he", "she"]:
                        corrected_words.append("was")   # "I was to..."
                    else:
                        corrected_words.append("water") # Default to water only in appropriate contexts
                else:
                    # Use the general context analysis
                    expanded = self.analyze_context(text, word.lower(), i)
                    corrected_words.append(expanded)
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)

    def is_improvement(self, original: str, corrected: str) -> bool:
        """Check if correction is actually an improvement"""
        # Don't accept corrections that create obviously wrong patterns
        wrong_patterns = [
            r'\b(I|you|he|she|we|they)\s+water\s+to\b',  # "I water to..."
            r'\b(I|you|he|she|we|they)\s+food\s+to\b',   # "I food to..."
            r'\bwater\s+to\s+(swim|go|come)\b',           # "water to swim"
        ]
        
        for pattern in wrong_patterns:
            if re.search(pattern, corrected, re.IGNORECASE):
                return False
        
        return True

    def basic_grammar_check(self, text: str) -> bool:
        """Basic grammar check as fallback"""
        # Simple patterns that indicate grammatical issues
        invalid_patterns = [
            r'\b(I|you|he|she|we|they)\s+(water|food|medicine)\s+to\b',
            r'\bwater\s+to\s+\w+',
            r'\bfood\s+to\s+\w+',
        ]
        
        for pattern in invalid_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False
        
        return True

    def validate_grammar(self, text: str) -> Tuple[bool, List[str]]:
        """Validate if sentence is grammatically correct"""
        issues = []
        
        # Check against invalid patterns
        for pattern in self.invalid_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(f"Invalid grammatical pattern detected")
        
        # Check for basic sentence structure
        if 'spacy' in st.session_state.models:
            try:
                doc = st.session_state.models['spacy'](text)
                
                # Check for subject-verb agreement
                subjects = [token for token in doc if token.dep_ == "nsubj"]
                verbs = [token for token in doc if token.pos_ == "VERB"]
                
                if subjects and verbs:
                    for subj in subjects:
                        for verb in verbs:
                            if subj.head == verb:
                                # Check agreement
                                if (subj.text.lower() == "i" and verb.text.endswith("s") and 
                                    verb.text not in ["is", "was", "has"]):
                                    issues.append(f"Subject-verb disagreement: '{subj.text}' with '{verb.text}'")
            except Exception as e:
                st.warning(f"Advanced grammar analysis failed: {e}")
        
        return len(issues) == 0, issues

    def generate_alternatives(self, text: str) -> List[str]:
        """Generate grammatically correct alternatives"""
        alternatives = []
        
        # Pattern-based alternatives for common cases
        text_lower = text.lower()
        
        if "wa to swi" in text_lower:
            alternatives.extend([
                "I want to swim",
                "I want to switch",
                "I was to swim"
            ])
        elif "wa to" in text_lower:
            alternatives.extend([
                text.replace("wa", "want"),
                text.replace("wa", "was")
            ])
        elif "nee hel" in text_lower:
            alternatives.extend([
                text.replace("nee hel", "need help"),
                text.replace("nee hel", "need help from")
            ])
        
        # Use grammar model for additional alternatives
        if 'grammar_corrector' in st.session_state.models:
            try:
                result = st.session_state.models['grammar_corrector'](
                    f"correct this sentence: {text}",
                    max_length=100,
                    num_return_sequences=2,
                    temperature=0.8
                )
                
                for alt in result:
                    alt_text = alt['generated_text'].strip()
                    # Only add if it's grammatically valid
                    if (alt_text != text and 
                        alt_text not in alternatives and 
                        self.basic_grammar_check(alt_text)):
                        alternatives.append(alt_text)
                        
            except Exception as e:
                st.warning(f"Alternative generation failed: {e}")
        
        # Format all alternatives properly
        formatted_alternatives = []
        for alt in alternatives:
            formatted_alt = self.format_sentence(alt)
            if formatted_alt not in formatted_alternatives:
                formatted_alternatives.append(formatted_alt)
        
        return formatted_alternatives[:3]  # Return top 3 alternatives

    def comprehensive_correction(self, text: str) -> Dict:
        """Apply comprehensive correction with validation"""
        results = {
            'original': text,
            'steps': [],
            'final': text,
            'is_grammatically_correct': False,  # Initialize with default value
            'grammar_issues': [],
            'alternatives': [],
            'confidence_score': 0
        }
        
        current_text = text
        
        # Step 1: Context-aware expansion (FIXED)
        expanded_text = self.context_aware_expansion(current_text)
        current_text = expanded_text
        results['steps'].append(('Context-Aware Expansion', current_text))
        
        # Step 2: Grammar correction with better handling
        if 'grammar_corrector' in st.session_state.models:
            try:
                grammar_result = st.session_state.models['grammar_corrector'](
                    f"grammar: {current_text}",
                    max_length=100,
                    temperature=0.3,  # Lower temperature for more consistent results
                    num_return_sequences=1
                )
                grammar_corrected = grammar_result[0]['generated_text'].strip()
                
                # Only use grammar correction if it actually improves the sentence
                if self.is_improvement(current_text, grammar_corrected):
                    current_text = grammar_corrected
                
                results['steps'].append(('Grammar Correction', current_text))
            except Exception as e:
                st.warning(f"Grammar correction failed: {e}")
        
        # Step 3: Final formatting
        final_text = self.format_sentence(current_text)
        results['final'] = final_text
        results['steps'].append(('Final Formatting', final_text))
        
        # Step 4: Grammar validation (ENSURE THIS ALWAYS RUNS)
        try:
            is_valid, issues = self.validate_grammar(final_text)
            results['is_grammatically_correct'] = is_valid
            results['grammar_issues'] = issues
        except Exception as e:
            # Fallback validation if spacy fails
            results['is_grammatically_correct'] = self.basic_grammar_check(final_text)
            results['grammar_issues'] = [f"Advanced grammar check failed: {e}"]
        
        # Step 5: Generate alternatives if not valid OR if low confidence
        if not results['is_grammatically_correct'] or results['confidence_score'] < 0.7:
            results['alternatives'] = self.generate_alternatives(text)
        
        # Calculate confidence (ENSURE THIS ALWAYS RUNS)
        results['confidence_score'] = self.calculate_confidence(
            text, 
            final_text, 
            results['is_grammatically_correct']
        )
        
        return results

    def format_sentence(self, text: str) -> str:
        """Format sentence properly"""
        if not text:
            return text
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        if text:
            text = text[0].upper() + text[1:]
        
        if text and not text[-1] in '.!?':
            if any(text.lower().startswith(word) for word in ['what', 'where', 'when', 'how', 'who', 'why']):
                text += '?'
            else:
                text += '.'
        
        return text

    def calculate_confidence(self, original: str, corrected: str, is_valid: bool) -> float:
        """Calculate confidence score"""
        base_score = 0.5
        
        if is_valid:
            base_score += 0.3
        
        # Length improvement
        if len(corrected) > len(original):
            base_score += 0.1
        
        # Word count improvement
        if len(corrected.split()) > len(original.split()):
            base_score += 0.1
        
        return min(1.0, base_score)

@st.cache_resource
def initialize_context_aware_system():
    """Initialize the context-aware system"""
    return ContextAwareSpeechCorrector()

def enhanced_speech_recognition(audio_file) -> Tuple[Optional[str], Optional[str]]:
    """Enhanced speech recognition"""
    try:
        recognizer = sr.Recognizer()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.read())
        
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        
        with sr.AudioFile(tmp_file.name) as source:
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            audio = recognizer.record(source)
        
        os.unlink(tmp_file.name)
        
        try:
            text = recognizer.recognize_google(audio, language='en-US')
            return text, None
        except sr.UnknownValueError:
            return None, "Could not understand the audio clearly"
        except sr.RequestError as e:
            return None, f"Recognition service error: {str(e)}"
            
    except Exception as e:
        return None, f"Audio processing error: {str(e)}"

def enhanced_text_to_speech(text: str, settings: Dict = None):
    """Enhanced TTS with quality settings"""
    try:
        engine = pyttsx3.init()
        
        if settings:
            engine.setProperty('rate', settings.get('rate', 150))
            engine.setProperty('volume', settings.get('volume', 0.9))
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            engine.save_to_file(text, tmp_file.name)
            engine.runAndWait()
            
            with open(tmp_file.name, 'rb') as f:
                audio_data = f.read()
            
            os.unlink(tmp_file.name)
            return audio_data
            
    except Exception as e:
        st.error(f"TTS error: {str(e)}")
        return None

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Context-Aware Speech Correction</h1>
        <p>Intelligent Disambiguation with Grammar Validation</p>
        <small>Advanced AI Models for Speech-Impaired Users</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize system
    if 'models' not in st.session_state:
        with st.spinner("🔄 Loading context-aware models..."):
            corrector = initialize_context_aware_system()
            st.session_state.models = corrector.load_models()
            st.session_state.corrector = corrector
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 System Status")
        
        # Model status
        models_status = {
            'spacy': '🔤 spaCy Linguistic Analysis',
            'grammar_corrector': '📝 T5 Grammar Correction', 
            'context_analyzer': '🧠 BERT Context Analysis'
        }
        
        for model_key, description in models_status.items():
            if model_key in st.session_state.models:
                st.success(f"✅ {description}")
            else:
                st.error(f"❌ {description}")
        
        st.markdown("---")
        
        # Settings
        st.subheader("⚙️ Voice Settings")
        voice_rate = st.slider("Speech Rate", 100, 200, 150)
        voice_volume = st.slider("Volume", 0.0, 1.0, 0.9)
        voice_settings = {'rate': voice_rate, 'volume': voice_volume}
        
        # Statistics
        st.subheader("📊 Session Stats")
        if 'total_corrections' not in st.session_state:
            st.session_state.total_corrections = 0
        
        st.metric("Corrections Made", st.session_state.total_corrections)
        
        # Help section
        st.subheader("💡 Examples")
        st.markdown("""
        **Try these ambiguous inputs:**
        - "I wa to swi" → "I want to swim"
        - "nee hel doc" → "Need help doctor"
        - "ea foo an drin wa" → "Eat food and drink water"
        """)
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input")
        
        # Input tabs
        tab1, tab2 = st.tabs(["✏️ Text Input", "🎵 Audio Upload"])
        
        user_input = ""
        
        with tab1:
            user_input = st.text_area(
                "Enter ambiguous speech text:",
                height=120,
                placeholder="Try these examples:\n• I wa to swi\n• I nee hel doc\n• ea foo an drin wa\n• I wa go swi",
                help="Enter speech that may have multiple interpretations"
            )
            
            # Test examples
            test_cases = [
                "I wa to swi",
                "nee hel doc",
                "ea foo an drin wa",
                "I wa go swi",
                "pai in nee"
            ]
            
            st.markdown("**🧪 Quick Test Cases:**")
            cols = st.columns(len(test_cases))
            
            for i, test_case in enumerate(test_cases):
                if cols[i].button(f'"{test_case}"', key=f"test_{i}", help=f"Test: {test_case}"):
                    user_input = test_case
                    st.session_state.test_input = test_case
        
        with tab2:
            audio_file = st.file_uploader(
                "Upload audio file",
                type=['wav', 'mp3', 'ogg', 'm4a'],
                help="Upload audio containing ambiguous speech"
            )
            
            if audio_file:
                st.audio(audio_file, format='audio/wav')
                
                if st.button("🎯 Process Audio", type="primary"):
                    with st.spinner("🔄 Processing audio with AI..."):
                        recognized_text, error = enhanced_speech_recognition(audio_file)
                    
                    if recognized_text:
                        user_input = recognized_text
                        st.success(f"✅ Recognized: '{recognized_text}'")
                    else:
                        st.error(f"❌ {error}")
        
        # Handle test input
        if hasattr(st.session_state, 'test_input'):
            user_input = st.session_state.test_input
            del st.session_state.test_input
        
        # Process button
        if user_input.strip() and st.button("🧠 Smart Correct", type="primary", use_container_width=True):
            with st.spinner("🔄 Analyzing context and validating grammar..."):
                corrector = st.session_state.corrector
                results = corrector.comprehensive_correction(user_input)
                st.session_state.results = results
                st.session_state.total_corrections += 1
    
    with col2:
        st.header("🎯 Smart Results")
        
        if 'results' not in st.session_state:
            st.info("👈 Enter ambiguous text and click 'Smart Correct'")
            
            # Show system capabilities
            with st.expander("🧠 System Capabilities"):
                st.markdown("""
                **Advanced Features:**
                - **Context Analysis**: Disambiguates words like "wa" → want/was/water
                - **Grammar Validation**: Ensures output is grammatically correct
                - **Alternative Generation**: Provides multiple correct options
                - **Confidence Scoring**: Measures correction reliability
                - **Pattern Recognition**: Learns from speech impairment patterns
                """)
        else:
            results = st.session_state.results
            
            # Show correction steps
            st.markdown("### 🔄 Processing Pipeline")
            for i, (step_name, step_result) in enumerate(results['steps']):
                st.markdown(f"**{i+1}. {step_name}:**")
                st.markdown(f'<div class="processing-step">{step_result}</div>', unsafe_allow_html=True)
            
            # Grammar validation result
            if results['is_grammatically_correct']:
                st.markdown("### ✅ Final Result (Grammar Validated)")
                st.markdown(f"""
                <div class="grammar-valid">
                    <h3 style="color: #059669; margin: 0;">{results['final']}</h3>
                    <div style="margin-top: 10px;">
                        <span style="background: #10b981; color: white; padding: 0.2rem 0.5rem; border-radius: 15px; font-size: 0.8rem;">
                            ✅ Grammatically Correct | Confidence: {results['confidence_score']:.0%}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("### ❌ Result (Grammar Issues Detected)")
                st.markdown(f"""
                <div class="grammar-invalid">
                    <h3 style="color: #dc2626; margin: 0;">{results['final']}</h3>
                    <div style="margin-top: 10px;">
                        <span style="background: #dc2626; color: white; padding: 0.2rem 0.5rem; border-radius: 15px; font-size: 0.8rem;">
                            ❌ Grammar Issues Found
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show grammar issues
                if results['grammar_issues']:
                    st.markdown("**📝 Grammar Issues:**")
                    for issue in results['grammar_issues']:
                        st.error(f"⚠️ {issue}")
                
                # Show alternatives
                if results['alternatives']:
                    st.markdown("### 💡 Recommended Alternatives")
                    for i, alternative in enumerate(results['alternatives']):
                        st.markdown(f"""
                        <div class="alternative-suggestion">
                            <strong>Option {i+1}:</strong> {alternative}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Validate each alternative
                        is_valid, _ = st.session_state.corrector.validate_grammar(alternative)
                        if is_valid:
                            st.success(f"✅ Option {i+1} is grammatically correct!")
                        else:
                            st.warning(f"⚠️ Option {i+1} may have grammar issues")
            
            # Action buttons
            st.markdown("### 🎛️ Actions")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("🔊 Play Audio", use_container_width=True):
                    # Use best alternative if original has issues
                    text_to_speak = results['final']
                    if not results['is_grammatically_correct'] and results['alternatives']:
                        text_to_speak = results['alternatives'][0]
                    
                    audio_data = enhanced_text_to_speech(text_to_speak, voice_settings)
                    if audio_data:
                        st.audio(audio_data, format='audio/wav')
            
            with col2:
                if st.button("📋 Copy Best", use_container_width=True):
                    best_text = results['final']
                    if not results['is_grammatically_correct'] and results['alternatives']:
                        best_text = results['alternatives'][0]
                    st.code(best_text)
                    st.success("✅ Best result ready to copy!")
            
            with col3:
                if st.button("📊 Details", use_container_width=True):
                    show_detailed_analysis(results)
            
            with col4:
                if st.button("🔄 Reset", use_container_width=True):
                    del st.session_state.results
                    st.rerun()

def show_detailed_analysis(results: Dict):
    """Show detailed analysis in expander"""
    with st.expander("🔍 Detailed Analysis", expanded=False):
        # Confidence breakdown
        st.markdown("**Confidence Analysis:**")
        st.progress(results['confidence_score'])
        
        # Original vs Final comparison
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original:**")
            st.code(results['original'])
        with col2:
            st.markdown("**Final:**")
            st.code(results['final'])
        
        # Processing steps
        st.markdown("**Processing Steps:**")
        for step_name, step_result in results['steps']:
            st.text(f"{step_name}: {step_result}")
        
        # Grammar analysis
        if results['grammar_issues']:
            st.markdown("**Grammar Issues Found:**")
            for issue in results['grammar_issues']:
                st.text(f"• {issue}")

if __name__ == "__main__":
    main()
