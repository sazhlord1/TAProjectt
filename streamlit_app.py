"""
Enhanced Multi-Technology Assessment Tutor
==========================================
An educational web application for teaching technology assessment and strategy.
Designed for university courses on technology assessment and technology strategy.

This app teaches students how to evaluate and compare up to 20 candidate technologies
using a structured, multi-criteria framework including:
- Technological Maturity
- Market Attractiveness (now with Financial Analysis)
- Organizational Capability & Strategic Fit
- Environment / ESG
- Technology Life Cycle & Diffusion (NEW)

Author: Educational Technology Assessment Tool
Framework: Streamlit + Plotly + NumPy + Pandas
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# =============================================================================
# CONSTANTS
# =============================================================================
MAX_TECH = 20

# Default weights for each index in the final synthesis
DEFAULT_WEIGHTS = {
    'maturity': 0.25,
    'market': 0.25,
    'capability': 0.25,
    'environment': 0.25
}

# Strategic scenario presets
STRATEGIC_SCENARIOS = {
    'Balanced': {'maturity': 0.25, 'market': 0.25, 'capability': 0.25, 'environment': 0.25},
    'Growth-Focused': {'maturity': 0.20, 'market': 0.40, 'capability': 0.25, 'environment': 0.15},
    'Risk-Averse': {'maturity': 0.35, 'market': 0.15, 'capability': 0.20, 'environment': 0.30},
    'Capability-Constrained': {'maturity': 0.20, 'market': 0.20, 'capability': 0.45, 'environment': 0.15},
    'ESG-Priority': {'maturity': 0.15, 'market': 0.20, 'capability': 0.20, 'environment': 0.45},
}

# Life Cycle Stages
LIFECYCLE_STAGES = [
    "Emerging / Introduction",
    "Growth",
    "Shake-out",
    "Maturity",
    "Decline"
]

LIFECYCLE_STAGE_SCORES = {
    "Emerging / Introduction": 50,
    "Growth": 80,
    "Shake-out": 60,
    "Maturity": 50,
    "Decline": 20
}

# =============================================================================
# STATE INITIALIZATION
# =============================================================================
def ensure_state():
    """Initialize session_state with all required data structures."""
    if 'initialized' not in st.session_state:
        st.session_state['initialized'] = True
        st.session_state['num_tech'] = 3
        
        # Technology metadata
        st.session_state['tech_names'] = [f"Technology {i+1}" for i in range(MAX_TECH)]
        st.session_state['tech_sectors'] = ["" for _ in range(MAX_TECH)]
        st.session_state['tech_descs'] = ["" for _ in range(MAX_TECH)]
        
        # Maturity parameters
        st.session_state['maturity_params'] = {
            'TRL': np.full(MAX_TECH, 5, dtype=int),
            'Stability': np.full(MAX_TECH, 5.0),
            'Complexity': np.full(MAX_TECH, 5.0),
            'IP': np.full(MAX_TECH, 5.0),
        }
        st.session_state['MaturityIndex'] = np.full(MAX_TECH, np.nan)
        
        # Market parameters
        st.session_state['market_params'] = {
            'TAM': np.full(MAX_TECH, 100.0),
            'CAGR': np.full(MAX_TECH, 10.0),
            'RelShare': np.full(MAX_TECH, 1.0),
            'Barriers': np.full(MAX_TECH, 5.0),
        }
        st.session_state['MarketIndex'] = np.full(MAX_TECH, np.nan)
        
        # Financial parameters
        st.session_state['financial_params'] = {
            'InitialInvestment': np.full(MAX_TECH, 10.0),
            'AnnualCashflow': np.full(MAX_TECH, 3.0),
            'TimeHorizon': np.full(MAX_TECH, 5, dtype=int),
            'DiscountRate': np.full(MAX_TECH, 10.0),
            'RiskScore': np.full(MAX_TECH, 5.0),
        }
        st.session_state['FinancialIndex'] = np.full(MAX_TECH, np.nan)
        st.session_state['EnhancedMarketIndex'] = np.full(MAX_TECH, np.nan)
        
        # Capability parameters
        st.session_state['cap_params'] = {
            'RD': np.full(MAX_TECH, 5.0),
            'Implementation': np.full(MAX_TECH, 5.0),
            'Partners': np.full(MAX_TECH, 5.0),
            'Finance': np.full(MAX_TECH, 5.0),
            'Fit': np.full(MAX_TECH, 5.0),
        }
        st.session_state['CapabilityIndex'] = np.full(MAX_TECH, np.nan)
        
        # Environment/ESG parameters
        st.session_state['env_params'] = {
            'Env': np.full(MAX_TECH, 5.0),
            'Soc': np.full(MAX_TECH, 5.0),
            'Reg': np.full(MAX_TECH, 5.0),
        }
        st.session_state['EnvironmentIndex'] = np.full(MAX_TECH, np.nan)
        
        # Life Cycle parameters
        st.session_state['lifecycle_params'] = {
            'Stage': ["Emerging / Introduction" for _ in range(MAX_TECH)],
            'YearsSinceLaunch': np.full(MAX_TECH, 2, dtype=int),
            'AdoptionLevel': np.full(MAX_TECH, 10.0),
            'ExpectedGrowth': np.full(MAX_TECH, 5.0),
            'CompetitiveIntensity': np.full(MAX_TECH, 5.0),
        }
        st.session_state['LifeCycleIndex'] = np.full(MAX_TECH, np.nan)
        
        # Synthesis weights
        st.session_state['weights'] = DEFAULT_WEIGHTS.copy()
        st.session_state['use_enhanced_market'] = False
        
        # Overall scores
        st.session_state['OverallScore'] = np.full(MAX_TECH, np.nan)


def get_active_tech_count():
    """Return the number of active technologies being assessed."""
    return st.session_state['num_tech']


def get_tech_name(i: int) -> str:
    """Get the name of technology i (0-indexed)."""
    return st.session_state['tech_names'][i]


# =============================================================================
# COMPUTATION FUNCTIONS
# =============================================================================

def compute_maturity_index(j):
    """Compute the Maturity Index for technology j."""
    params = st.session_state['maturity_params']
    
    S_TRL = (params['TRL'][j] / 9.0) * 100
    S_stab = params['Stability'][j] * 10
    S_int = (10 - params['Complexity'][j]) * 10
    S_IP = params['IP'][j] * 10
    
    M_j = 0.40 * S_TRL + 0.25 * S_stab + 0.20 * S_int + 0.15 * S_IP
    return M_j


def compute_market_index(j):
    """Compute the Market Attractiveness Index for technology j."""
    params = st.session_state['market_params']
    
    TAM = params['TAM'][j]
    CAGR = params['CAGR'][j]
    RelShare = params['RelShare'][j]
    Barriers = params['Barriers'][j]
    
    # TAM score
    if TAM < 50:
        S_TAM = 20
    elif TAM <= 200:
        S_TAM = 20 + 60 * (TAM - 50) / 150
    else:
        S_TAM = 80
    
    # CAGR score
    if CAGR < 5:
        S_CAGR = 30
    elif CAGR <= 15:
        S_CAGR = 30 + 30 * (CAGR - 5) / 10
    else:
        S_CAGR = 90
    
    # Relative share score
    S_share = min(50 * RelShare, 100)
    
    # Barrier score (inverted)
    S_barrier = 10 * (10 - Barriers)
    
    A_j = 0.30 * S_TAM + 0.30 * S_CAGR + 0.25 * S_share + 0.15 * S_barrier
    return A_j


def compute_financial_metrics(j):
    """Compute financial metrics for technology j."""
    params = st.session_state['financial_params']
    
    invest = params['InitialInvestment'][j]
    cf = params['AnnualCashflow'][j]
    horizon = params['TimeHorizon'][j]
    discount = params['DiscountRate'][j] / 100.0
    
    # ROI
    if invest == 0:
        roi = 0
    else:
        roi = ((cf * horizon - invest) / invest) * 100
    
    # Payback
    if cf <= 0:
        payback = float('inf')
    else:
        payback = invest / cf
    
    # NPV
    if discount == 0:
        npv = cf * horizon - invest
    else:
        npv_cf = sum([cf / ((1 + discount) ** t) for t in range(1, horizon + 1)])
        npv = npv_cf - invest
    
    return roi, payback, npv


def compute_financial_index(j, npv_max=None):
    """Compute the Financial Attractiveness Index for technology j."""
    roi, payback, npv = compute_financial_metrics(j)
    
    # ROI score
    if roi <= 0:
        S_ROI = 20
    elif roi <= 50:
        S_ROI = 20 + 60 * (roi / 50)
    else:
        S_ROI = 80
    
    # Payback score
    if payback <= 3:
        S_PB = 80
    elif payback <= 10:
        S_PB = 80 - 40 * ((payback - 3) / 7)
    else:
        S_PB = 20
    
    # NPV score
    if npv_max is None or npv_max <= 0:
        npv_max = 50.0  # Default normalization
    
    if npv <= 0:
        S_NPV = 20
    elif npv <= npv_max:
        S_NPV = 20 + 60 * (npv / npv_max)
    else:
        S_NPV = 80
    
    Financial_j = 0.40 * S_ROI + 0.30 * S_PB + 0.30 * S_NPV
    return Financial_j


def compute_enhanced_market_index(j):
    """Compute Enhanced Market Index combining market and financial aspects."""
    A_j = st.session_state['MarketIndex'][j]
    if np.isnan(A_j):
        A_j = 0
    
    Financial_j = st.session_state['FinancialIndex'][j]
    if np.isnan(Financial_j):
        Financial_j = 0
    
    # Risk penalty
    risk = st.session_state['financial_params']['RiskScore'][j]
    S_risk = (10 - risk) * 10
    
    Enhanced_j = 0.60 * A_j + 0.25 * Financial_j + 0.15 * S_risk
    return Enhanced_j


def compute_capability_index(j):
    """Compute the Capability & Strategic Fit Index for technology j."""
    params = st.session_state['cap_params']
    
    scores = [
        params['RD'][j],
        params['Implementation'][j],
        params['Partners'][j],
        params['Finance'][j],
        params['Fit'][j]
    ]
    
    C_j = np.mean(scores) * 10
    return C_j


def compute_environment_index(j):
    """Compute the Environment & ESG Index for technology j."""
    params = st.session_state['env_params']
    
    S_env = params['Env'][j] * 10
    S_soc = params['Soc'][j] * 10
    S_reg = params['Reg'][j] * 10
    
    E_j = 0.40 * S_env + 0.30 * S_soc + 0.30 * S_reg
    return E_j


def compute_lifecycle_index(j):
    """Compute Life Cycle Index for technology j."""
    params = st.session_state['lifecycle_params']
    
    # Stage score
    stage = params['Stage'][j]
    S_stage = LIFECYCLE_STAGE_SCORES.get(stage, 50)
    
    # Adoption score (peaks around 50%)
    adoption = params['AdoptionLevel'][j]
    S_adopt = 100 - abs(adoption - 50)
    
    # Growth score
    growth = params['ExpectedGrowth'][j]
    S_growth = growth * 10
    
    LC_j = 0.40 * S_stage + 0.30 * S_adopt + 0.30 * S_growth
    return LC_j


def get_lifecycle_strategy(j):
    """Derive qualitative strategy label based on life cycle stage and index."""
    params = st.session_state['lifecycle_params']
    stage = params['Stage'][j]
    lc_index = st.session_state['LifeCycleIndex'][j]
    
    if np.isnan(lc_index):
        return "Not calculated"
    
    if stage in ["Emerging / Introduction", "Growth"] and lc_index > 60:
        return "🚀 Invest to build position"
    elif stage == "Maturity" and lc_index > 50:
        return "🛡️ Defend and optimize"
    elif stage == "Maturity" and lc_index <= 50:
        return "💰 Harvest / milk cash"
    elif stage == "Decline":
        return "📉 Divest or reposition"
    elif stage == "Shake-out":
        return "⚔️ Compete aggressively or exit"
    else:
        return "🤔 Monitor closely"


def compute_overall_score(j, weights, use_enhanced_market=False):
    """Compute the Overall Score for technology j using weighted sum."""
    M = st.session_state['MaturityIndex'][j]
    
    if use_enhanced_market:
        A = st.session_state['EnhancedMarketIndex'][j]
    else:
        A = st.session_state['MarketIndex'][j]
    
    C = st.session_state['CapabilityIndex'][j]
    E = st.session_state['EnvironmentIndex'][j]
    
    # Handle missing values
    M = 0 if np.isnan(M) else M
    A = 0 if np.isnan(A) else A
    C = 0 if np.isnan(C) else C
    E = 0 if np.isnan(E) else E
    
    overall = (weights['maturity'] * M + 
               weights['market'] * A + 
               weights['capability'] * C + 
               weights['environment'] * E)
    
    return overall


def normalize_weights(w_m, w_a, w_c, w_e):
    """Normalize weights to sum to 1."""
    total = w_m + w_a + w_c + w_e
    if total == 0:
        return 0.25, 0.25, 0.25, 0.25
    return w_m / total, w_a / total, w_c / total, w_e / total


@st.cache_data
def load_technologies():
    return pd.read_csv("Technologies.csv")

tech_df = load_technologies()

# =============================================================================
# PAGE FUNCTIONS
# =============================================================================
if "techs_initialized" not in st.session_state:
    st.session_state["techs_initialized"] = True

    for i in range(len(tech_df)):
        st.session_state["tech_names"][i] = tech_df.iloc[i]["Technology"]
        st.session_state["tech_sectors"][i] = tech_df.iloc[i]["Sector"]
        st.session_state["tech_descs"][i] = tech_df.iloc[i]["Description"]

def page_define_techs():
    """Page 1: Define Technologies"""
    st.header("📋 Step 1 of 7: Define Technologies")
    
    # Learning Objectives
    with st.expander("🎯 Learning Objectives", expanded=False):
        st.markdown("""
        By completing this section, you will:
        - Learn the importance of clearly defining technology scope before assessment
        - Practice creating distinguishing identifiers for technologies
        - Understand how sector context affects technology evaluation
        - Develop skills in articulating technology value propositions concisely
        """)
    
    st.markdown("""
    Welcome to the **Multi-Technology Assessment Tutor**! This tool helps you evaluate 
    and compare up to 20 candidate technologies for investment decisions.
    
    **Start by defining the technologies you want to assess.**
    """)
    
    with st.expander("📚 Educational Note: Why Define Technologies First?", expanded=False):
        st.markdown("""
        Before diving into quantitative assessment, it's crucial to clearly define what you're evaluating:
        
        - **Technology Name**: A clear, distinguishing identifier that stakeholders will recognize
        - **Sector/Domain**: The industry or application area (e.g., Energy, Healthcare, AI, Manufacturing)
        - **Description**: Brief explanation of what the technology does and its key value proposition
        
        **Why This Matters:**
        - Ensures consistency throughout your assessment
        - Helps communicate your analysis to stakeholders
        - Prevents scope creep and keeps evaluation focused
        - Facilitates comparison across similar technologies
        
        **Best Practice:** Define technologies at the same level of abstraction. For example, compare 
        "Solar PV" with "Wind Turbines" rather than mixing "Renewable Energy" with "Wind Turbines."
        """)
    
    # Number of technologies selector
    st.subheader("Number of Technologies")
    num_tech = st.slider(
        "How many technologies do you want to assess?",
        min_value=1,
        max_value=min(MAX_TECH, len(tech_df))
        value=st.session_state['num_tech'],
        help="You can assess between 1 and 20 technologies"
    )
    st.session_state['num_tech'] = num_tech
    
    st.divider()
    
    # Technology definition inputs
    st.subheader("Define Each Technology")
    
    cols = st.columns(2)
    
    for i in range(num_tech):
        col_idx = i % 2
        with cols[col_idx]:
            with st.expander(f"🔧 Technology {i+1}: {st.session_state['tech_names'][i]}", expanded=(i < 3)):
                name = st.text_input(
                    "Name",
                    value=st.session_state['tech_names'][i],
                    key=f"name_{i}",
                    help="Give this technology a clear, distinguishing name"
                )
                st.session_state['tech_names'][i] = name
                
                sector = st.text_input(
                    "Sector/Domain",
                    value=st.session_state['tech_sectors'][i],
                    key=f"sector_{i}",
                    help="E.g., Energy, Healthcare, AI, Manufacturing"
                )
                st.session_state['tech_sectors'][i] = sector
                
                desc = st.text_area(
                    "Description",
                    value=st.session_state['tech_descs'][i],
                    key=f"desc_{i}",
                    height=80,
                    help="Brief description of what this technology does"
                )
                st.session_state['tech_descs'][i] = desc
    
    st.divider()
    
    # Summary table
    st.subheader("📊 Technology Summary")
    summary_data = {
        'Technology': [st.session_state['tech_names'][i] for i in range(num_tech)],
        'Sector': [st.session_state['tech_sectors'][i] for i in range(num_tech)],
        'Description': [
            st.session_state['tech_descs'][i][:50] + '...' 
            if len(st.session_state['tech_descs'][i]) > 50 
            else st.session_state['tech_descs'][i] 
            for i in range(num_tech)
        ]
    }
    df_summary = pd.DataFrame(summary_data)
    df_summary.index = range(1, num_tech + 1)
    df_summary.index.name = '#'
    st.dataframe(df_summary, use_container_width=True)
    
    # Reflection Questions
    st.markdown("### 🤔 Reflection Questions")
    st.markdown("""
    - Are your technology definitions at a consistent level of detail and abstraction?
    - Do the names clearly distinguish each technology from the others?
    - Have you captured the key value proposition in each description?
    """)
    
    st.info("✅ **Next Step**: Navigate to 'Maturity Assessment' in the sidebar to evaluate technological readiness.")


def page_maturity():
    """Page 2: Maturity Assessment"""
    st.header("🔬 Step 2 of 7: Maturity Assessment")
    
    num_tech = get_active_tech_count()
    
    # Learning Objectives
    with st.expander("🎯 Learning Objectives", expanded=False):
        st.markdown("""
        By completing this section, you will:
        - Understand the TRL (Technology Readiness Level) framework and its 9 levels
        - Learn how to translate qualitative maturity judgments into numeric indices
        - Distinguish between technical maturity and commercial readiness
        - Recognize the importance of integration complexity and IP protection
        - Practice assessing multiple dimensions of technological maturity
        """)
    
    with st.expander("📚 Educational: Understanding the Maturity Index", expanded=True):
        st.markdown("""
        **Technological Maturity** measures how ready a technology is for commercial deployment.
        
        **The Four Dimensions:**
        
        1. **TRL (Technology Readiness Level)**: NASA's 1-9 scale measuring development stage  
           - TRL 1-3: Research/concept phase (basic principles, concept formulation)  
           - TRL 4-6: Development/demonstration (lab validation, prototype testing)  
           - TRL 7-9: Deployment/commercialization ready (system proven, operational)
        
        2. **Stability**: How reliable and consistent is the prototype/product?  
           - Measures repeatability, failure rates, and performance consistency  
           - Critical for manufacturing scale-up and customer acceptance
        
        3. **Integration Complexity**: How difficult is it to integrate with existing systems?  
           - Note: Higher scores mean **more** complexity (worse)  
           - Includes compatibility issues, infrastructure requirements, training needs
        
        4. **IP Strength & Novelty**: How strong is the intellectual property position?  
           - Patents, trade secrets, proprietary know-how  
           - Freedom to operate without infringing others' IP
        
        **Weights:**
        - TRL: 40% (fundamental readiness)
        - Stability: 25%
        - Integration Complexity: 20% (penalty)
        - IP: 15%
        """)
        
        st.markdown("**Maturity Index Formula:**")
        st.latex(r"M_j = 0.40 \cdot S_{TRL,j} + 0.25 \cdot S_{stab,j} + 0.20 \cdot S_{int,j} + 0.15 \cdot S_{IP,j}")
        
        st.markdown("**Sub-scores (0–100 scale):**")
        col1, col2 = st.columns(2)
        with col1:
            st.latex(r"S_{TRL,j} = \frac{TRL_j}{9} \times 100")
            st.latex(r"S_{stab,j} = Stability_j \times 10")
        with col2:
            st.latex(r"S_{int,j} = (10 - Complexity_j) \times 10")
            st.latex(r"S_{IP,j} = IP_j \times 10")
    
    # Mini-Case Example
    with st.expander("💡 Mini-Case Example", expanded=False):
        st.markdown("""
        **Case: AI-Based Medical Diagnostic Tool**
        
        Imagine assessing an AI system that analyzes medical images to detect diseases.
        
        **Assessment:**
        - TRL = 7: System demonstrated in operational environment (hospital pilot)  
        - Stability = 7.5: 85% accuracy consistently achieved across test cases  
        - Integration Complexity = 6: Moderate – requires new IT infrastructure and staff training  
        - IP Strength = 8: Strong patents on algorithm and database
        
        **Calculated Maturity Index:**
        - S_TRL = (7/9) × 100 ≈ 77.8  
        - S_stab = 7.5 × 10 = 75.0  
        - S_int = (10−6) × 10 = 40.0  
        - S_IP = 8 × 10 = 80.0
        
        **M ≈ 0.40(77.8) + 0.25(75.0) + 0.20(40.0) + 0.15(80.0) ≈ 68.1**
        
        **Interpretation:** Moderately mature technology. While technically proven and well-protected, 
        integration challenges could slow adoption. Consider investing in integration support tools.
        """)
    
    st.divider()
    
    # Input section
    st.subheader("📝 Enter Maturity Parameters")
    
    params = st.session_state['maturity_params']
    
    for i in range(num_tech):
        with st.expander(f"🔧 {get_tech_name(i)}", expanded=(i < 2)):
            col1, col2 = st.columns(2)
            
            with col1:
                trl = st.slider(
                    "TRL (Technology Readiness Level)",
                    min_value=1, max_value=9,
                    value=int(params['TRL'][i]),
                    key=f"trl_{i}",
                    help="1=Basic research, 9=Fully commercialized"
                )
                params['TRL'][i] = trl
                
                stability = st.slider(
                    "Stability (Prototype Reliability)",
                    min_value=0.0, max_value=10.0,
                    value=float(params['Stability'][i]),
                    step=0.5,
                    key=f"stab_{i}",
                    help="0=Very unreliable, 10=Highly reliable"
                )
                params['Stability'][i] = stability
            
            with col2:
                complexity = st.slider(
                    "Integration Complexity",
                    min_value=0.0, max_value=10.0,
                    value=float(params['Complexity'][i]),
                    step=0.5,
                    key=f"comp_{i}",
                    help="0=Easy to integrate, 10=Very complex (worse)"
                )
                params['Complexity'][i] = complexity
                
                ip = st.slider(
                    "IP Strength & Novelty",
                    min_value=0.0, max_value=10.0,
                    value=float(params['IP'][i]),
                    step=0.5,
                    key=f"ip_{i}",
                    help="0=No IP protection, 10=Strong patents & novelty"
                )
                params['IP'][i] = ip
    
    st.divider()
    
    # Calculate button
    if st.button("🧮 Calculate Maturity Indices for All Technologies", type="primary"):
        for j in range(num_tech):
            st.session_state['MaturityIndex'][j] = compute_maturity_index(j)
        st.success("✅ Maturity indices calculated successfully!")
    
    # Results display
    maturity_scores = st.session_state['MaturityIndex'][:num_tech]
    
    if not np.all(np.isnan(maturity_scores)):
        st.subheader("📊 Results: Maturity Index")
        
        # Results table
        results_data = {
            'Technology': [get_tech_name(i) for i in range(num_tech)],
            'TRL': [st.session_state['maturity_params']['TRL'][i] for i in range(num_tech)],
            'Stability': [st.session_state['maturity_params']['Stability'][i] for i in range(num_tech)],
            'Complexity': [st.session_state['maturity_params']['Complexity'][i] for i in range(num_tech)],
            'IP Strength': [st.session_state['maturity_params']['IP'][i] for i in range(num_tech)],
            'Maturity Index': [
                f"{maturity_scores[i]:.1f}" if not np.isnan(maturity_scores[i]) else "N/A"
                for i in range(num_tech)
            ]
        }
        df_results = pd.DataFrame(results_data)
        st.dataframe(df_results, use_container_width=True)
        
        # Bar chart
        fig = px.bar(
            x=[get_tech_name(i) for i in range(num_tech)],
            y=maturity_scores,
            labels={'x': 'Technology', 'y': 'Maturity Index (0–100)'},
            title='Maturity Index by Technology',
            color=maturity_scores,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Reflection Questions
    st.markdown("### 🤔 Reflection Questions")
    st.markdown("""
    - Which technologies have high TRL but low overall maturity? What does this suggest?
    - How might integration complexity affect your deployment timeline and costs?
    - If you had to improve one dimension for each technology, which would it be and why?
    - Do any technologies show a mismatch between technical maturity and IP protection?
    """)
    
    st.info("✅ **Next Step**: Navigate to 'Market & Financial Assessment' to evaluate commercial attractiveness and financial viability.")


def page_market():
    """Page 3: Market Attractiveness & Financial Assessment"""
    st.header("📈 Step 3 of 7: Market Attractiveness & Financial Assessment")
    
    num_tech = get_active_tech_count()
    
    # Learning Objectives
    with st.expander("🎯 Learning Objectives", expanded=False):
        st.markdown("""
        By completing this section, you will:
        - Understand how to evaluate total addressable market (TAM) and growth rates
        - Learn to assess competitive position through relative market share
        - Master simple financial metrics: ROI, Payback Period, and NPV
        - Recognize the role of risk and uncertainty in market evaluation
        - Practice integrating market and financial analysis for technology decisions
        """)
    
    with st.expander("📚 Educational: Understanding Market Attractiveness", expanded=True):
        st.markdown("""
        **Market Attractiveness** evaluates the commercial potential of a technology.
        
        **Market Dimensions:**
        
        1. **TAM (Total Addressable Market)**: The total market size in millions of dollars  
           Larger markets offer more opportunity but may attract more competition.
        
        2. **CAGR (Compound Annual Growth Rate)**: Expected market growth rate (%)  
           High CAGR (>15%) suggests fast-growing, potentially disruptive markets.
        
        3. **Relative Market Share**: Your share divided by main competitor's share  
           >1 = leader; <1 = challenger; affects pricing power and profitability.
        
        4. **Adoption Barriers**: Obstacles to market entry and diffusion  
           Regulatory, technical, behavioral, switching costs, standards, etc.
        
        **Financial Dimensions (NEW):**
        
        1. **ROI (Return on Investment)**  
           (Total cash in − Investment) / Investment × 100%.  
           Higher is better; compare to opportunity cost of capital.
        
        2. **Payback Period**  
           Years to recover initial investment.  
           Shorter payback reduces exposure to uncertainty.
        
        3. **NPV (Net Present Value)**  
           Present value of future cash flows minus investment, discounted by cost of capital.
        
        4. **Risk Score** (0–10)  
           Combined view of market, competitive, and execution risk; used as a penalty.
        """)
        
        st.markdown("**Basic Market Index Formula:**")
        st.latex(r"A_j = 0.30 \cdot S_{TAM,j} + 0.30 \cdot S_{CAGR,j} + 0.25 \cdot S_{share,j} + 0.15 \cdot S_{barrier,j}")
        
        st.markdown("**Enhanced Market Index (Market + Finance + Risk):**")
        st.latex(r"Enhanced_j = 0.60 \cdot A_j + 0.25 \cdot Financial_j + 0.15 \cdot (10 - Risk_j) \times 10")
        
        st.markdown("**Financial Index:**")
        st.latex(r"Financial_j = 0.40 \cdot S_{ROI} + 0.30 \cdot S_{Payback} + 0.30 \cdot S_{NPV}")
    
    # Mini-Case Example
    with st.expander("💡 Mini-Case Example", expanded=False):
        st.markdown("""
        **Case: Electric Vehicle Battery Technology**
        
        **Market Parameters:**
        - TAM: $180M  
        - CAGR: 18%  
        - Relative Share: 0.8  
        - Barriers: 7 (high)
        
        **Financial Parameters:**
        - Initial Investment: $25M  
        - Annual Net Cash Flow: $8M  
        - Time Horizon: 5 years  
        - Discount Rate: 12%
        
        The tool converts these into:
        - Market Index (TAM, CAGR, share, barriers)
        - Financial Index (ROI, Payback, NPV)
        - Enhanced Market Index = Market + Finance − Risk penalty
        
        Interpretation: Good growth market with reasonable financials, but high barriers and 
        competitive challenges reduce attractiveness. Strategic partnerships or staged investment 
        might be appropriate.
        """)
    
    st.divider()
    
    # Input section
    st.subheader("📝 Enter Market Parameters")
    
    market_params = st.session_state['market_params']
    fin_params = st.session_state['financial_params']
    
    for i in range(num_tech):
        with st.expander(f"📊 {get_tech_name(i)}", expanded=(i < 2)):
            st.markdown("##### Market Characteristics")
            col1, col2 = st.columns(2)
            
            with col1:
                tam = st.number_input(
                    "TAM (Total Addressable Market) in $M",
                    min_value=0.0, max_value=10000.0,
                    value=float(market_params['TAM'][i]),
                    step=10.0,
                    key=f"tam_{i}",
                    help="Total market size in millions of dollars"
                )
                market_params['TAM'][i] = tam
                
                cagr = st.number_input(
                    "CAGR (%) - Expected Growth Rate",
                    min_value=-50.0, max_value=100.0,
                    value=float(market_params['CAGR'][i]),
                    step=0.5,
                    key=f"cagr_{i}",
                    help="Annual growth rate (can be negative for declining markets)"
                )
                market_params['CAGR'][i] = cagr
            
            with col2:
                rel_share = st.number_input(
                    "Relative Market Share (your / competitor's)",
                    min_value=0.0, max_value=10.0,
                    value=float(market_params['RelShare'][i]),
                    step=0.1,
                    key=f"share_{i}",
                    help="Ratio of your market share to main competitor (1.0 = equal)"
                )
                market_params['RelShare'][i] = rel_share
                
                barriers = st.slider(
                    "Adoption Barriers",
                    min_value=0.0, max_value=10.0,
                    value=float(market_params['Barriers'][i]),
                    step=0.5,
                    key=f"barrier_{i}",
                    help="0=No barriers, 10=Very high barriers (worse)"
                )
                market_params['Barriers'][i] = barriers
            
            st.markdown("---")
            st.markdown("##### Financial & Risk Parameters")
            
            col3, col4 = st.columns(2)
            
            with col3:
                invest = st.number_input(
                    "Initial Investment (Capex) in $M",
                    min_value=0.0, max_value=1000.0,
                    value=float(fin_params['InitialInvestment'][i]),
                    step=1.0,
                    key=f"invest_{i}",
                    help="Upfront capital investment required"
                )
                fin_params['InitialInvestment'][i] = invest
                
                cashflow = st.number_input(
                    "Annual Net Cash Flow (average) in $M",
                    min_value=-100.0, max_value=500.0,
                    value=float(fin_params['AnnualCashflow'][i]),
                    step=0.5,
                    key=f"cf_{i}",
                    help="Average annual cash flow over time horizon"
                )
                fin_params['AnnualCashflow'][i] = cashflow
                
                horizon = st.number_input(
                    "Time Horizon (years)",
                    min_value=1, max_value=20,
                    value=int(fin_params['TimeHorizon'][i]),
                    key=f"horizon_{i}",
                    help="Investment evaluation period"
                )
                fin_params['TimeHorizon'][i] = horizon
            
            with col4:
                discount = st.number_input(
                    "Discount Rate (%)",
                    min_value=0.0, max_value=50.0,
                    value=float(fin_params['DiscountRate'][i]),
                    step=0.5,
                    key=f"discount_{i}",
                    help="Cost of capital for NPV calculation"
                )
                fin_params['DiscountRate'][i] = discount
                
                risk = st.slider(
                    "Perceived Market Risk",
                    min_value=0.0, max_value=10.0,
                    value=float(fin_params['RiskScore'][i]),
                    step=0.5,
                    key=f"risk_{i}",
                    help="0=Very low risk, 10=Very high risk"
                )
                fin_params['RiskScore'][i] = risk
    
    st.divider()
    
    # Calculate button
    if st.button("🧮 Calculate Market & Financial Indices for All Technologies", type="primary"):
        # Calculate basic market index
        for j in range(num_tech):
            st.session_state['MarketIndex'][j] = compute_market_index(j)
        
        # Calculate financial index
        # First find max NPV for normalization
        npvs = []
        for j in range(num_tech):
            _, _, npv = compute_financial_metrics(j)
            npvs.append(npv if npv > 0 else 0)
        npv_max = max(npvs) if npvs else 50.0
        
        for j in range(num_tech):
            st.session_state['FinancialIndex'][j] = compute_financial_index(j, npv_max)
        
        # Calculate enhanced market index
        for j in range(num_tech):
            st.session_state['EnhancedMarketIndex'][j] = compute_enhanced_market_index(j)
        
        st.success("✅ Market and financial indices calculated successfully!")
    
    # Results display
    market_scores = st.session_state['MarketIndex'][:num_tech]
    financial_scores = st.session_state['FinancialIndex'][:num_tech]
    enhanced_scores = st.session_state['EnhancedMarketIndex'][:num_tech]
    
    if not np.all(np.isnan(market_scores)):
        st.subheader("📊 Results: Market & Financial Analysis")
        
        # Compute financial metrics for display
        fin_metrics = []
        for i in range(num_tech):
            roi, payback, npv = compute_financial_metrics(i)
            fin_metrics.append({
                'ROI': f"{roi:.1f}%" if roi != float('inf') else "N/A",
                'Payback': f"{payback:.1f}y" if payback < 50 else ">50y",
                'NPV': f"${npv:.1f}M"
            })
        
        # Results table
        results_data = {
            'Technology': [get_tech_name(i) for i in range(num_tech)],
            'TAM ($M)': [f"{market_params['TAM'][i]:.0f}" for i in range(num_tech)],
            'CAGR (%)': [f"{market_params['CAGR'][i]:.1f}" for i in range(num_tech)],
            'ROI': [fin_metrics[i]['ROI'] for i in range(num_tech)],
            'Payback': [fin_metrics[i]['Payback'] for i in range(num_tech)],
            'NPV': [fin_metrics[i]['NPV'] for i in range(num_tech)],
            'Risk': [f"{fin_params['RiskScore'][i]:.1f}" for i in range(num_tech)],
            'Market Index': [
                f"{market_scores[i]:.1f}" if not np.isnan(market_scores[i]) else "N/A"
                for i in range(num_tech)
            ],
            'Financial Index': [
                f"{financial_scores[i]:.1f}" if not np.isnan(financial_scores[i]) else "N/A"
                for i in range(num_tech)
            ],
            'Enhanced Index': [
                f"{enhanced_scores[i]:.1f}" if not np.isnan(enhanced_scores[i]) else "N/A"
                for i in range(num_tech)
            ]
        }
        df_results = pd.DataFrame(results_data)
        st.dataframe(df_results, use_container_width=True)
        
        # Visualization tabs
        tab1, tab2, tab3 = st.tabs(["Market Index", "Financial Index", "Enhanced Index"])
        
        with tab1:
            fig1 = px.bar(
                x=[get_tech_name(i) for i in range(num_tech)],
                y=market_scores,
                labels={'x': 'Technology', 'y': 'Market Index (0–100)'},
                title='Basic Market Attractiveness Index',
                color=market_scores,
                color_continuous_scale='RdYlGn'
            )
            fig1.update_layout(showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
        
        with tab2:
            fig2 = px.bar(
                x=[get_tech_name(i) for i in range(num_tech)],
                y=financial_scores,
                labels={'x': 'Technology', 'y': 'Financial Index (0–100)'},
                title='Financial Attractiveness Index',
                color=financial_scores,
                color_continuous_scale='Blues'
            )
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
        
        with tab3:
            fig3 = px.bar(
                x=[get_tech_name(i) for i in range(num_tech)],
                y=enhanced_scores,
                labels={'x': 'Technology', 'y': 'Enhanced Market Index (0–100)'},
                title='Enhanced Market Index (Market + Finance + Risk)',
                color=enhanced_scores,
                color_continuous_scale='Viridis'
            )
            fig3.update_layout(showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)
    
    # Reflection Questions
    st.markdown("### 🤔 Reflection Questions")
    st.markdown("""
    - Which technologies have strong market metrics but weak financial returns? What might explain this?
    - How sensitive are your NPV calculations to the discount rate? Try varying it by ±5%.
    - Do high-risk technologies have correspondingly high potential returns (ROI)?
    - Which matters more for your organization: short payback period or high total NPV?
    - How does the Enhanced Market Index change your ranking compared to the basic Market Index?
    """)
    
    st.info("✅ **Next Step**: Navigate to 'Capability Assessment' to evaluate organizational readiness.")


def page_capability():
    """Page 4: Capability & Strategic Fit Assessment"""
    st.header("🏢 Step 4 of 7: Capability & Strategic Fit Assessment")
    
    num_tech = get_active_tech_count()
    
    # Learning Objectives
    with st.expander("🎯 Learning Objectives", expanded=False):
        st.markdown("""
        By completing this section, you will:
        - Understand the gap between technology potential and organizational capability
        - Learn to assess internal R&D and implementation capacity
        - Recognize the importance of ecosystem partnerships
        - Evaluate strategic fit with organizational goals and resources
        - Distinguish between "can we do this?" and "should we do this?"
        """)
    
    with st.expander("📚 Educational: Understanding Capability & Strategic Fit", expanded=True):
        st.markdown("""
        **Capability & Strategic Fit** evaluates whether your organization can successfully 
        develop, deploy, and commercialize a technology.
        
        **The Five Capability Dimensions (0–10 scale each):**
        
        1. **R&D Capability**  
           Technical expertise, research capacity, access to labs and tools.
        
        2. **Implementation / Manufacturing**  
           Ability to produce, deploy, and scale at required quality and cost.
        
        3. **Partner & Ecosystem Access**  
           Relationships with suppliers, distributors, universities, standards bodies.
        
        4. **Financial Capacity**  
           Ability to fund development and deployment through uncertainty.
        
        5. **Strategic Fit**  
           Alignment with core business, strategy, and leadership priorities.
        
        All dimensions are weighted equally because missing *any one* can create a serious gap:
        
        """)
        st.latex(r"C_j = \frac{1}{5} \sum_{k=1}^{5} Cap_{k,j} \times 10")
        st.markdown("""
        This is simply the average of five 0–10 scores, scaled to 0–100.
        """)
    
    # Mini-Case Example
    with st.expander("💡 Mini-Case Example", expanded=False):
        st.markdown("""
        **Case: Quantum Computing for Drug Discovery**
        
        A pharmaceutical company evaluates a quantum-computing application for molecular simulation.
        
        **Assessment (0–10 scale):**
        - R&D Capability = 4 (limited quantum expertise)
        - Implementation = 3 (no quantum infrastructure)
        - Partners = 7 (good ecosystem ties)
        - Finance = 8 (budget available)
        - Strategic Fit = 9 (strong alignment with drug discovery strategy)
        
        **Capability Index:**
        
        ```text
        C = (4 + 3 + 7 + 8 + 9) / 5 × 10 = 62.0
        ```
        
        Interpretation: Moderate capability despite strong strategic fit and financing.  
        Weak R&D and implementation scores signal high execution risk. A partnership-heavy 
        strategy is more realistic than fully in-house development.
        """)
    
    st.divider()
    
    # Input section
    st.subheader("📝 Enter Capability Parameters")
    
    params = st.session_state['cap_params']
    
    for i in range(num_tech):
        with st.expander(f"🏢 {get_tech_name(i)}", expanded=(i < 2)):
            col1, col2 = st.columns(2)
            
            with col1:
                rd = st.slider(
                    "R&D Capability",
                    min_value=0.0, max_value=10.0,
                    value=float(params['RD'][i]),
                    step=0.5,
                    key=f"rd_{i}",
                    help="0 = No R&D capacity, 10 = World-class R&D"
                )
                params['RD'][i] = rd
                
                impl = st.slider(
                    "Implementation / Manufacturing Capability",
                    min_value=0.0, max_value=10.0,
                    value=float(params['Implementation'][i]),
                    step=0.5,
                    key=f"impl_{i}",
                    help="0 = No production capacity, 10 = Excellent manufacturing / deployment capability"
                )
                params['Implementation'][i] = impl
                
                partners = st.slider(
                    "Partner & Ecosystem Access",
                    min_value=0.0, max_value=10.0,
                    value=float(params['Partners'][i]),
                    step=0.5,
                    key=f"partner_{i}",
                    help="0 = No partnerships, 10 = Strong ecosystem relationships"
                )
                params['Partners'][i] = partners
            
            with col2:
                finance = st.slider(
                    "Financial Capacity",
                    min_value=0.0, max_value=10.0,
                    value=float(params['Finance'][i]),
                    step=0.5,
                    key=f"fin_{i}",
                    help="0 = No budget, 10 = Ample funding available"
                )
                params['Finance'][i] = finance
                
                fit = st.slider(
                    "Strategic Fit",
                    min_value=0.0, max_value=10.0,
                    value=float(params['Fit'][i]),
                    step=0.5,
                    key=f"fit_{i}",
                    help="0 = Does not fit current strategy, 10 = Perfect strategic alignment"
                )
                params['Fit'][i] = fit
    
    st.divider()
    
    # Calculate button
    if st.button("🧮 Calculate Capability Indices for All Technologies", type="primary"):
        for j in range(num_tech):
            st.session_state['CapabilityIndex'][j] = compute_capability_index(j)
        st.success("✅ Capability indices calculated successfully!")
    
    # Results display
    cap_scores = st.session_state['CapabilityIndex'][:num_tech]
    
    if not np.all(np.isnan(cap_scores)):
        st.subheader("📊 Results: Capability & Strategic Fit Index")
        
        # Results table
        results_data = {
            'Technology': [get_tech_name(i) for i in range(num_tech)],
            'R&D': [f"{params['RD'][i]:.1f}" for i in range(num_tech)],
            'Implementation': [f"{params['Implementation'][i]:.1f}" for i in range(num_tech)],
            'Partners': [f"{params['Partners'][i]:.1f}" for i in range(num_tech)],
            'Finance': [f"{params['Finance'][i]:.1f}" for i in range(num_tech)],
            'Strategic Fit': [f"{params['Fit'][i]:.1f}" for i in range(num_tech)],
            'Capability Index': [
                f"{cap_scores[i]:.1f}" if not np.isnan(cap_scores[i]) else "N/A"
                for i in range(num_tech)
            ]
        }
        df_results = pd.DataFrame(results_data)
        st.dataframe(df_results, use_container_width=True)
        
        # Bar chart
        fig = px.bar(
            x=[get_tech_name(i) for i in range(num_tech)],
            y=cap_scores,
            labels={'x': 'Technology', 'y': 'Capability Index (0–100)'},
            title='Capability & Strategic Fit Index by Technology',
            color=cap_scores,
            color_continuous_scale='Blues'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Reflection Questions
    st.markdown("### 🤔 Reflection Questions")
    st.markdown("""
    - Which technologies have high market/maturity scores but low capability? What implementation risks arise?
    - For technologies with capability gaps, could partnerships or alliances close those gaps?
    - Are there technologies where strong strategic fit compensates for weaknesses in other capabilities?
    - Across your portfolio, which single capability dimension is the most common bottleneck?
    """)
    
    st.info("✅ **Next Step**: Navigate to 'Environment & ESG' to evaluate sustainability factors.")


def page_environment():
    """Page 5: Environment & ESG Assessment"""
    st.header("🌱 Step 5 of 7: Environment & ESG Assessment")
    
    num_tech = get_active_tech_count()
    
    # Learning Objectives
    with st.expander("🎯 Learning Objectives", expanded=False):
        st.markdown("""
        By completing this section, you will:
        - Understand the growing importance of ESG (Environmental, Social, Governance) factors
        - Learn to assess environmental benefits and risks of technologies
        - Evaluate social acceptance and ethical considerations
        - Recognize how regulatory trends affect technology viability
        - Practice integrating sustainability into technology strategy
        """)
    
    with st.expander("📚 Educational: Understanding Environment & ESG", expanded=True):
        st.markdown("""
        **Environment & ESG** evaluates the sustainability and societal impact of a technology.
        
        **The Three ESG Dimensions:**
        
        1. **Environmental Benefit**  
           Emissions reduction, resource efficiency, circularity, biodiversity protection.
        
        2. **Social / Ethical Acceptance**  
           Jobs, equity, privacy, trust, access, worker health & safety.
        
        3. **Regulatory Favorability**  
           Current compliance, policy support, incentives, future regulatory risk.
        """)
        
        st.markdown("**ESG Index Formula:**")
        st.latex(r"E_j = 0.40 \cdot S_{env,j} + 0.30 \cdot S_{soc,j} + 0.30 \cdot S_{reg,j}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.latex(r"S_{env,j} = Env_j \times 10")
        with col2:
            st.latex(r"S_{soc,j} = Soc_j \times 10")
        with col3:
            st.latex(r"S_{reg,j} = Reg_j \times 10")
    
    # Mini-Case Example
    with st.expander("💡 Mini-Case Example", expanded=False):
        st.markdown("""
        **Case: Advanced Coal with CCS vs Offshore Wind**
        
        Coal + CCS: Env=5, Soc=4, Reg=3 → ESG ≈ 41  
        Offshore Wind: Env=8, Soc=7, Reg=9 → ESG ≈ 80
        
        Even if both are technically feasible, ESG performance clearly favours offshore wind, 
        which will likely enjoy better long-term policy and capital access.
        """)
    
    st.divider()
    
    # Input section
    st.subheader("📝 Enter Environment & ESG Parameters")
    
    params = st.session_state['env_params']
    
    for i in range(num_tech):
        with st.expander(f"🌱 {get_tech_name(i)}", expanded=(i < 2)):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                env = st.slider(
                    "Environmental Benefit",
                    min_value=0.0, max_value=10.0,
                    value=float(params['Env'][i]),
                    step=0.5,
                    key=f"env_{i}",
                    help="0 = Harmful, 10 = Strong positive environmental impact"
                )
                params['Env'][i] = env
            
            with col2:
                soc = st.slider(
                    "Social / Ethical Acceptance",
                    min_value=0.0, max_value=10.0,
                    value=float(params['Soc'][i]),
                    step=0.5,
                    key=f"soc_{i}",
                    help="0 = Major social concerns, 10 = Broadly accepted"
                )
                params['Soc'][i] = soc
            
            with col3:
                reg = st.slider(
                    "Regulatory Favorability",
                    min_value=0.0, max_value=10.0,
                    value=float(params['Reg'][i]),
                    step=0.5,
                    key=f"reg_{i}",
                    help="0 = Hostile regulation, 10 = Strong regulatory support"
                )
                params['Reg'][i] = reg
    
    st.divider()
    
    # Calculate button
    if st.button("🧮 Calculate ESG Indices for All Technologies", type="primary"):
        for j in range(num_tech):
            st.session_state['EnvironmentIndex'][j] = compute_environment_index(j)
        st.success("✅ ESG indices calculated successfully!")
    
    # Results display
    env_scores = st.session_state['EnvironmentIndex'][:num_tech]
    
    if not np.all(np.isnan(env_scores)):
        st.subheader("📊 Results: Environment & ESG Index")
        
        results_data = {
            'Technology': [get_tech_name(i) for i in range(num_tech)],
            'Environmental': [f"{params['Env'][i]:.1f}" for i in range(num_tech)],
            'Social': [f"{params['Soc'][i]:.1f}" for i in range(num_tech)],
            'Regulatory': [f"{params['Reg'][i]:.1f}" for i in range(num_tech)],
            'ESG Index': [
                f"{env_scores[i]:.1f}" if not np.isnan(env_scores[i]) else "N/A"
                for i in range(num_tech)
            ]
        }
        df_results = pd.DataFrame(results_data)
        st.dataframe(df_results, use_container_width=True)
        
        fig = px.bar(
            x=[get_tech_name(i) for i in range(num_tech)],
            y=env_scores,
            labels={'x': 'Technology', 'y': 'ESG Index (0–100)'},
            title='Environment & ESG Index by Technology',
            color=env_scores,
            color_continuous_scale='Greens'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 🤔 Reflection Questions")
    st.markdown("""
    - Do any technologies have strong environmental benefits but weak social acceptance?
    - How might regulatory trends over the next 5–10 years change these scores?
    - Could design or business-model changes materially improve ESG scores?
    - If you had an ESG-only filter, which technologies would you immediately exclude?
    """)
    
    st.info("✅ **Next Step**: Navigate to 'Life Cycle Analysis' to understand adoption dynamics.")


def page_lifecycle():
    """Page 6: Technology Life Cycle & Diffusion"""
    st.header("🔄 Step 6 of 7: Technology Life Cycle & Diffusion")
    
    num_tech = get_active_tech_count()
    
    # Learning Objectives
    with st.expander("🎯 Learning Objectives", expanded=False):
        st.markdown("""
        By completing this section, you will:
        - Understand technology and industry life cycle stages
        - Learn to connect life cycle stage to appropriate strategic actions
        - Recognize how adoption dynamics affect risk and opportunity
        - Practice assessing maturity from a market/competitive perspective
        - Distinguish between technical maturity and market maturity
        """)
    
    with st.expander("📚 Educational: Understanding Technology Life Cycle", expanded=True):
        st.markdown("""
        **Technology Life Cycle** describes how technologies evolve through predictable stages:
        
        ```
        Emerging → Growth → Shake-out → Maturity → Decline
            ↑         ↑          ↑           ↑          ↑
          High risk  Fast      Consolidation Stable   Shrinking
          Low volume growth    Competition   Cash     market
        ```
        
        **Stage Characteristics (simplified scores):**
        - Emerging / Introduction (50): High uncertainty, low adoption; invest to explore.
        - Growth (80): Rapid adoption and scaling; invest aggressively.
        - Shake-out (60): Intense competition and consolidation; differentiate or exit.
        - Maturity (50): Stable market, incremental innovation; optimize and harvest.
        - Decline (20): Shrinking market; harvest or divest.
        
        Additional factors:
        - **Adoption Level (%)**: S-curve; scoring peaks around 50%.  
          \(S_{adopt} = 100 - |Adoption - 50|\)
        - **Expected Growth (0–10)**: Future adoption trajectory;  
          \(S_{growth} = ExpectedGrowth \times 10\)
        """)
        
        st.markdown("**Life Cycle Index Formula:**")
        st.latex(r"LC_j = 0.40 \cdot S_{stage} + 0.30 \cdot S_{adopt} + 0.30 \cdot S_{growth}")
    
    # Mini-Case Example
    with st.expander("💡 Mini-Case Example", expanded=False):
        st.markdown("""
        **Example: Three Battery Technologies**
        
        - Lithium-ion (Maturity): stable, high adoption, moderate growth → LC ≈ 50–55  
        - Solid-state (Emerging): low adoption, very high expected growth → LC ≈ 60+  
        - Lead-acid (Decline): shrinking market, low growth → LC ≈ 30
        
        Interpretation: Even with decent current adoption, a declining technology scores low on LC,
        while an emerging technology with strong growth expectations can score higher and justify
        forward-looking investment.
        """)
    
    st.divider()
    
    # Input section
    st.subheader("📝 Enter Life Cycle Parameters")
    
    params = st.session_state['lifecycle_params']
    
    for i in range(num_tech):
        with st.expander(f"🔄 {get_tech_name(i)}", expanded=(i < 2)):
            col1, col2 = st.columns(2)
            
            with col1:
                stage = st.selectbox(
                    "Technology Life Cycle Stage",
                    options=LIFECYCLE_STAGES,
                    index=LIFECYCLE_STAGES.index(params['Stage'][i]),
                    key=f"stage_{i}",
                    help="Select the current life cycle stage"
                )
                params['Stage'][i] = stage
                
                years = st.number_input(
                    "Years Since First Commercial Launch",
                    min_value=0, max_value=100,
                    value=int(params['YearsSinceLaunch'][i]),
                    key=f"years_{i}",
                    help="Approximate years since first commercial availability"
                )
                params['YearsSinceLaunch'][i] = years
                
                adoption = st.slider(
                    "Current Adoption Level (% of target market)",
                    min_value=0.0, max_value=100.0,
                    value=float(params['AdoptionLevel'][i]),
                    step=1.0,
                    key=f"adopt_{i}",
                    help="Current penetration in your target market"
                )
                params['AdoptionLevel'][i] = adoption
            
            with col2:
                growth = st.slider(
                    "Expected Adoption Growth",
                    min_value=0.0, max_value=10.0,
                    value=float(params['ExpectedGrowth'][i]),
                    step=0.5,
                    key=f"growth_{i}",
                    help="0 = Stagnant/declining, 10 = Explosive growth expected"
                )
                params['ExpectedGrowth'][i] = growth
                
                competition = st.slider(
                    "Competitive Intensity (optional)",
                    min_value=0.0, max_value=10.0,
                    value=float(params['CompetitiveIntensity'][i]),
                    step=0.5,
                    key=f"compete_{i}",
                    help="0 = Few competitors, 10 = Very crowded market"
                )
                params['CompetitiveIntensity'][i] = competition
    
    st.divider()
    
    # Calculate button
    if st.button("🧮 Calculate Life Cycle Indices for All Technologies", type="primary"):
        for j in range(num_tech):
            st.session_state['LifeCycleIndex'][j] = compute_lifecycle_index(j)
        st.success("✅ Life cycle indices calculated successfully!")
    
    # Results display
    lc_scores = st.session_state['LifeCycleIndex'][:num_tech]
    
    if not np.all(np.isnan(lc_scores)):
        st.subheader("📊 Results: Life Cycle Analysis")
        
        # Results table with strategy recommendations
        results_data = {
            'Technology': [get_tech_name(i) for i in range(num_tech)],
            'Stage': [params['Stage'][i] for i in range(num_tech)],
            'Years': [params['YearsSinceLaunch'][i] for i in range(num_tech)],
            'Adoption %': [f"{params['AdoptionLevel'][i]:.0f}%" for i in range(num_tech)],
            'Growth': [f"{params['ExpectedGrowth'][i]:.1f}" for i in range(num_tech)],
            'Competition': [f"{params['CompetitiveIntensity'][i]:.1f}" for i in range(num_tech)],
            'LC Index': [
                f"{lc_scores[i]:.1f}" if not np.isnan(lc_scores[i]) else "N/A"
                for i in range(num_tech)
            ],
            'Strategy': [get_lifecycle_strategy(i) for i in range(num_tech)]
        }
        df_results = pd.DataFrame(results_data)
        st.dataframe(df_results, use_container_width=True)
        
        # Visualizations
        tab1, tab2 = st.tabs(["Life Cycle Index", "Adoption vs Years"])
        
        with tab1:
            fig1 = px.bar(
                x=[get_tech_name(i) for i in range(num_tech)],
                y=lc_scores,
                labels={'x': 'Technology', 'y': 'Life Cycle Index (0–100)'},
                title='Life Cycle Index by Technology',
                color=lc_scores,
                color_continuous_scale='RdYlGn'
            )
            fig1.update_layout(showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
        
        with tab2:
            chart_data = pd.DataFrame({
                'Technology': [get_tech_name(i) for i in range(num_tech)],
                'Years': [params['YearsSinceLaunch'][i] for i in range(num_tech)],
                'Adoption': [params['AdoptionLevel'][i] for i in range(num_tech)],
                'Stage': [params['Stage'][i] for i in range(num_tech)],
                'LC Index': lc_scores
            })
            
            fig2 = px.scatter(
                chart_data,
                x='Years',
                y='Adoption',
                size='LC Index',
                color='Stage',
                hover_name='Technology',
                title='Technology Adoption Over Time',
                labels={'Years': 'Years Since Launch', 'Adoption': 'Adoption Level (%)'},
                size_max=30
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    # Reflection Questions
    st.markdown("### 🤔 Reflection Questions")
    st.markdown("""
    - Do life cycle stages align with your intuitive view of each technology?
    - Which emerging technologies deserve disproportionate investment?
    - Are there mature technologies you should harvest or divest?
    - How does life cycle information complement your maturity and market indices?
    """)
    
    st.info("✅ **Next Step**: Navigate to 'Synthesis & Ranking' for the final portfolio analysis!")


def page_synthesis():
    """Page 7: Synthesis & Portfolio Ranking"""
    st.header("🏆 Step 7 of 7: Synthesis & Portfolio Ranking")
    
    num_tech = get_active_tech_count()
    
    # Safety: ensure OverallScore is initialized
    if 'OverallScore' not in st.session_state:
        st.session_state['OverallScore'] = np.full(MAX_TECH, np.nan)
    
    # Learning Objectives
    with st.expander("🎯 Learning Objectives", expanded=False):
        st.markdown("""
        By completing this section, you will:
        - Learn to combine multiple criteria into overall technology rankings
        - Understand how weight assignments reflect strategic priorities
        - Practice sensitivity analysis by adjusting weights
        - Develop skills in portfolio visualization and interpretation
        - Learn to communicate technology assessment results to stakeholders
        """)
    
    with st.expander("📚 Educational: Weighted Sum Synthesis", expanded=True):
        st.markdown("""
        **Portfolio Synthesis** combines all four indices into an **Overall Score** 
        for each technology using a weighted sum approach (multi-criteria decision analysis).
        """)
        st.latex(r"Overall_j = w_M \cdot M_j + w_A \cdot A_j + w_C \cdot C_j + w_E \cdot E_j")
        st.markdown("""
        Where:
        - \(M_j\) = Maturity Index (technical readiness)  
        - \(A_j\) = Market Index (basic or enhanced with financials)  
        - \(C_j\) = Capability Index (organizational readiness)  
        - \(E_j\) = Environment/ESG Index (sustainability)  
        - \(w_M, w_A, w_C, w_E\) = Weights (normalized to sum to 1)
        
        **Strategic Scenarios:**
        - Balanced (25/25/25/25)  
        - Growth-Focused (emphasize Market)  
        - Risk-Averse (emphasize Maturity)  
        - Capability-Constrained (emphasize Capability)  
        - ESG-Priority (emphasize Environment)
        
        You can quickly explore these or configure your own.
        """)
        
        st.info("""
        Advanced MCDA methods (AHP, TOPSIS, ELECTRE, fuzzy MCDA) and portfolio optimization 
        exist, but a transparent weighted-sum approach is ideal for teaching and for 
        interactive "what-if" analysis in classroom settings.
        """)
    
    st.divider()
    
    # Current indices summary and missing indices
    st.subheader("📊 Current Index Values")
    
    missing_indices = []
    for i in range(num_tech):
        missing = []
        if np.isnan(st.session_state['MaturityIndex'][i]):
            missing.append('Maturity')
        if np.isnan(st.session_state['MarketIndex'][i]):
            missing.append('Market')
        if np.isnan(st.session_state['CapabilityIndex'][i]):
            missing.append('Capability')
        if np.isnan(st.session_state['EnvironmentIndex'][i]):
            missing.append('ESG')
        if missing:
            missing_indices.append((get_tech_name(i), missing))
    
    if missing_indices:
        st.warning("⚠️ Some indices are missing. Missing values will be treated as 0 in the overall score:")
        for tech_name, missing in missing_indices:
            st.write(f"- **{tech_name}**: Missing {', '.join(missing)}")
    
    # Market index choice
    st.markdown("**Market Dimension Configuration:**")
    use_enhanced = st.radio(
        "Which Market index would you like to use?",
        options=[
            "Basic Market Index (TAM, CAGR, share, barriers only)",
            "Enhanced Market Index (includes ROI, NPV, Payback, and risk)"
        ],
        index=1 if st.session_state.get('use_enhanced_market', False) else 0,
        help="Enhanced index provides more complete commercial viability assessment"
    )
    st.session_state['use_enhanced_market'] = "Enhanced" in use_enhanced
    
    # Summary table of all indices
    if st.session_state['use_enhanced_market']:
        market_col_name = 'Enhanced Market'
        market_values = [
            f"{st.session_state['EnhancedMarketIndex'][i]:.1f}" 
            if not np.isnan(st.session_state['EnhancedMarketIndex'][i]) else "⚠️ N/A"
            for i in range(num_tech)
        ]
    else:
        market_col_name = 'Market'
        market_values = [
            f"{st.session_state['MarketIndex'][i]:.1f}" 
            if not np.isnan(st.session_state['MarketIndex'][i]) else "⚠️ N/A"
            for i in range(num_tech)
        ]
    
    summary_data = {
        'Technology': [get_tech_name(i) for i in range(num_tech)],
        'Sector': [st.session_state['tech_sectors'][i] for i in range(num_tech)],
        'Maturity': [
            f"{st.session_state['MaturityIndex'][i]:.1f}" 
            if not np.isnan(st.session_state['MaturityIndex'][i]) else "⚠️ N/A"
            for i in range(num_tech)
        ],
        market_col_name: market_values,
        'Capability': [
            f"{st.session_state['CapabilityIndex'][i]:.1f}" 
            if not np.isnan(st.session_state['CapabilityIndex'][i]) else "⚠️ N/A"
            for i in range(num_tech)
        ],
        'ESG': [
            f"{st.session_state['EnvironmentIndex'][i]:.1f}" 
            if not np.isnan(st.session_state['EnvironmentIndex'][i]) else "⚠️ N/A"
            for i in range(num_tech)
        ],
        'Life Cycle': [
            f"{st.session_state['LifeCycleIndex'][i]:.1f}" 
            if not np.isnan(st.session_state['LifeCycleIndex'][i]) else "N/A"
            for i in range(num_tech)
        ]
    }
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, use_container_width=True)
    
    st.divider()
    
    # Weight configuration
    st.subheader("⚖️ Configure Synthesis Weights")
    
    # Quick presets
    st.markdown("**Quick Presets:**")
    preset_cols = st.columns(5)
    for idx, (scenario_name, scenario_weights) in enumerate(STRATEGIC_SCENARIOS.items()):
        with preset_cols[idx]:
            if st.button(scenario_name, key=f"preset_{scenario_name}"):
                st.session_state['weights'] = scenario_weights.copy()
                # Use new API if available, fall back to experimental for older versions
                if hasattr(st, "rerun"):
                    st.rerun()
                elif hasattr(st, "experimental_rerun"):
                    st.experimental_rerun()
    
    st.markdown("**Or adjust weights manually:**")
    
    weight_cols = st.columns(4)
    weights = st.session_state['weights']
    
    with weight_cols[0]:
        w_m = st.slider("Maturity Weight", 0.0, 1.0, weights['maturity'], 0.05, key="w_maturity")
    with weight_cols[1]:
        w_a = st.slider("Market Weight", 0.0, 1.0, weights['market'], 0.05, key="w_market")
    with weight_cols[2]:
        w_c = st.slider("Capability Weight", 0.0, 1.0, weights['capability'], 0.05, key="w_capability")
    with weight_cols[3]:
        w_e = st.slider("ESG Weight", 0.0, 1.0, weights['environment'], 0.05, key="w_environment")
    
    # Normalize weights
    w_m_norm, w_a_norm, w_c_norm, w_e_norm = normalize_weights(w_m, w_a, w_c, w_e)
    
    st.markdown(f"""
    **Normalized Weights** (sum to 1.0):  
    - Maturity: **{w_m_norm:.2f}** | Market: **{w_a_norm:.2f}** | Capability: **{w_c_norm:.2f}** | ESG: **{w_e_norm:.2f}**
    """)
    
    st.session_state['weights'] = {
        'maturity': w_m_norm,
        'market': w_a_norm,
        'capability': w_c_norm,
        'environment': w_e_norm
    }
    
    st.divider()
    
    # Calculate overall scores
    if st.button("🏆 Compute Overall Scores & Ranking", type="primary"):
        for j in range(num_tech):
            st.session_state['OverallScore'][j] = compute_overall_score(
                j, 
                st.session_state['weights'],
                st.session_state['use_enhanced_market']
            )
        st.success("✅ Overall scores calculated! See ranking below.")
    
    overall_scores = st.session_state['OverallScore'][:num_tech]
    
    if not np.all(np.isnan(overall_scores)):
        st.subheader("🏆 Final Technology Ranking")
        
        # Build ranking dataframe
        ranking_data = []
        for i in range(num_tech):
            if st.session_state['use_enhanced_market']:
                market_val = st.session_state['EnhancedMarketIndex'][i] if not np.isnan(st.session_state['EnhancedMarketIndex'][i]) else 0
            else:
                market_val = st.session_state['MarketIndex'][i] if not np.isnan(st.session_state['MarketIndex'][i]) else 0
            
            ranking_data.append({
                'Technology': get_tech_name(i),
                'Sector': st.session_state['tech_sectors'][i],
                'Maturity': st.session_state['MaturityIndex'][i] if not np.isnan(st.session_state['MaturityIndex'][i]) else 0,
                'Market': market_val,
                'Capability': st.session_state['CapabilityIndex'][i] if not np.isnan(st.session_state['CapabilityIndex'][i]) else 0,
                'ESG': st.session_state['EnvironmentIndex'][i] if not np.isnan(st.session_state['EnvironmentIndex'][i]) else 0,
                'Life Cycle': st.session_state['LifeCycleIndex'][i] if not np.isnan(st.session_state['LifeCycleIndex'][i]) else 0,
                'Overall Score': overall_scores[i]
            })
        
        df_ranking = pd.DataFrame(ranking_data)
        df_ranking = df_ranking.sort_values('Overall Score', ascending=False).reset_index(drop=True)
        df_ranking.insert(0, 'Rank', range(1, len(df_ranking) + 1))
        
        # Format numeric columns
        for col in ['Maturity', 'Market', 'Capability', 'ESG', 'Life Cycle', 'Overall Score']:
            df_ranking[col] = df_ranking[col].apply(lambda x: f"{x:.1f}")
        
        st.dataframe(df_ranking, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Visualizations
        st.subheader("📊 Portfolio Visualizations")
        
        # Rebuild with numeric values for charts
        chart_data = []
        for i in range(num_tech):
            if st.session_state['use_enhanced_market']:
                market_val = st.session_state['EnhancedMarketIndex'][i] if not np.isnan(st.session_state['EnhancedMarketIndex'][i]) else 0
            else:
                market_val = st.session_state['MarketIndex'][i] if not np.isnan(st.session_state['MarketIndex'][i]) else 0
            
            chart_data.append({
                'Technology': get_tech_name(i),
                'Sector': st.session_state['tech_sectors'][i],
                'Maturity': st.session_state['MaturityIndex'][i] if not np.isnan(st.session_state['MaturityIndex'][i]) else 0,
                'Market': market_val,
                'Capability': st.session_state['CapabilityIndex'][i] if not np.isnan(st.session_state['CapabilityIndex'][i]) else 0,
                'ESG': st.session_state['EnvironmentIndex'][i] if not np.isnan(st.session_state['EnvironmentIndex'][i]) else 0,
                'Life Cycle': st.session_state['LifeCycleIndex'][i] if not np.isnan(st.session_state['LifeCycleIndex'][i]) else 0,
                'Overall Score': overall_scores[i],
                'LC Stage': st.session_state['lifecycle_params']['Stage'][i]
            })
        df_chart = pd.DataFrame(chart_data)
        df_chart = df_chart.sort_values('Overall Score', ascending=False)
        
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "Bar Chart", 
            "Scatter Plot", 
            "Radar Chart",
            "Life Cycle View"
        ])
        
        with viz_tab1:
            st.markdown("**Overall Score Ranking**")
            fig_bar = px.bar(
                df_chart,
                x='Technology',
                y='Overall Score',
                color='Overall Score',
                color_continuous_scale='Viridis',
                title='Technology Ranking by Overall Score'
            )
            fig_bar.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with viz_tab2:
            st.markdown("**Market vs Maturity Portfolio Map**")
            st.markdown("*Bubble size = Overall Score, Color = Overall Score*")
            
            fig_scatter = px.scatter(
                df_chart,
                x='Maturity',
                y='Market',
                size='Overall Score',
                color='Overall Score',
                hover_name='Technology',
                hover_data=['Capability', 'ESG', 'Sector', 'Life Cycle'],
                color_continuous_scale='RdYlGn',
                title='Technology Portfolio: Market Attractiveness vs Maturity',
                size_max=40
            )
            fig_scatter.update_layout(
                xaxis_title='Maturity Index',
                yaxis_title='Market Attractiveness Index'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.info("""
            Interpretation:
            - Top-right: high market & high maturity → prime investment candidates  
            - Top-left: high market, low maturity → high potential but risky  
            - Bottom-right: low market, high maturity → niche / legacy opportunities  
            - Bottom-left: low on both → avoid or divest
            """)
        
        with viz_tab3:
            st.markdown("**Multi-Dimensional Radar Comparison**")
            
            sorted_techs = df_chart.sort_values('Overall Score', ascending=False)['Technology'].tolist()
            default_selection = sorted_techs[:min(3, num_tech)]
            
            selected_techs = st.multiselect(
                "Select technologies to compare:",
                options=[get_tech_name(i) for i in range(num_tech)],
                default=default_selection,
                max_selections=6
            )
            
            if selected_techs:
                categories = ['Maturity', 'Market', 'Capability', 'ESG']
                
                fig_radar = go.Figure()
                colors = px.colors.qualitative.Set1
                
                for idx, tech in enumerate(selected_techs):
                    tech_row = df_chart[df_chart['Technology'] == tech].iloc[0]
                    values = [tech_row[cat] for cat in categories]
                    values.append(values[0])  # close polygon
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories + [categories[0]],
                        fill='toself',
                        name=tech,
                        line_color=colors[idx % len(colors)],
                        fillcolor=colors[idx % len(colors)],
                        opacity=0.3
                    ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )
                    ),
                    showlegend=True,
                    title='Technology Comparison Radar Chart'
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.info("Select at least one technology to display the radar chart.")
        
        with viz_tab4:
            st.markdown("**Life Cycle Stage Analysis**")
            st.markdown("*Shows how technologies distribute across life cycle stages*")
            
            fig_lc = px.scatter(
                df_chart,
                x='Life Cycle',
                y='Overall Score',
                color='LC Stage',
                size='Market',
                hover_name='Technology',
                hover_data=['Maturity', 'Capability', 'ESG'],
                title='Life Cycle Maturity vs Overall Score',
                size_max=30
            )
            fig_lc.update_layout(
                xaxis_title='Life Cycle Index',
                yaxis_title='Overall Score'
            )
            st.plotly_chart(fig_lc, use_container_width=True)
            
            st.markdown("**Life Cycle Stage Distribution:**")
            stage_counts = df_chart['LC Stage'].value_counts()
            fig_stage = px.bar(
                x=stage_counts.index,
                y=stage_counts.values,
                labels={'x': 'Stage', 'y': 'Number of Technologies'},
                title='Distribution of Technologies by Life Cycle Stage'
            )
            st.plotly_chart(fig_stage, use_container_width=True)
        
        st.divider()
        
        # Reflection prompts
        st.subheader("🤔 Reflection Questions for Students")
        st.markdown("""
        **General Analysis**
        1. Do the top-ranked technologies match your intuition? Why or why not?  
        2. Which technologies have high Market & Maturity but low Capability? What does this imply?
        
        **Financial Analysis**
        3. How does using the Enhanced Market Index change rankings?  
        4. Are there technologies with strong ROI but high risk scores? How should you treat them?
        
        **Life Cycle Insights**
        5. Is your portfolio balanced across life cycle stages?  
        6. Are there emerging technologies that deserve more investment based on their overall scores?
        
        **Strategy**
        7. Are there "hidden gems" – strong on multiple dimensions, but not rank #1?  
        8. How sensitive is your ranking to weight changes? A robust ranking should be reasonably stable.
        """)
        
        st.divider()
        
        # Export option
        st.subheader("📥 Export Results")
        
        export_df = df_ranking.copy()
        csv = export_df.to_csv(index=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📄 Download Ranking as CSV",
                data=csv,
                file_name="technology_ranking.csv",
                mime="text/csv"
            )
        with col2:
            st.info("CSV includes all indices and overall scores.")
        
        st.markdown("---")
        st.markdown("""
        **Future Extensions (for advanced users):**
        - Monte Carlo simulation for uncertainty analysis  
        - Real options valuation for staged investments  
        - Portfolio optimization under budget constraints  
        - More advanced MCDA methods and group decision-making  
        - Integration with external patent/market/funding databases
        """)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    
    st.set_page_config(
        page_title="Enhanced Multi-Technology Assessment Tutor",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize state
    ensure_state()
    
    # Sidebar navigation
    st.sidebar.title("🔬 Tech Assessment Tutor")
    st.sidebar.markdown("*Enhanced with Financial & Life Cycle Analysis*")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigate to:",
        [
            "1️⃣ Define Technologies",
            "2️⃣ Maturity Assessment",
            "3️⃣ Market & Financial Assessment",
            "4️⃣ Capability Assessment",
            "5️⃣ Environment & ESG",
            "6️⃣ Life Cycle Analysis",
            "7️⃣ Synthesis & Ranking",
        ]
    )

    # Routing
    if page.startswith("1️⃣"):
        page_define_techs()
    elif page.startswith("2️⃣"):
        page_maturity()
    elif page.startswith("3️⃣"):
        page_market()
    elif page.startswith("4️⃣"):
        page_capability()
    elif page.startswith("5️⃣"):
        page_environment()
    elif page.startswith("6️⃣"):
        page_lifecycle()
    elif page.startswith("7️⃣"):
        page_synthesis()
    
    st.sidebar.markdown("---")
    
    # Quick status in sidebar
    num_tech = get_active_tech_count()
    st.sidebar.subheader("📊 Progress Status")
    st.sidebar.write(f"Technologies defined: **{num_tech}**")
    
    mat_done = int(np.sum(~np.isnan(st.session_state['MaturityIndex'][:num_tech])))
    mkt_done = int(np.sum(~np.isnan(st.session_state['MarketIndex'][:num_tech])))
    cap_done = int(np.sum(~np.isnan(st.session_state['CapabilityIndex'][:num_tech])))
    env_done = int(np.sum(~np.isnan(st.session_state['EnvironmentIndex'][:num_tech])))
    lc_done = int(np.sum(~np.isnan(st.session_state['LifeCycleIndex'][:num_tech])))
    
    st.sidebar.write(f"Maturity: {mat_done}/{num_tech}")
    st.sidebar.write(f"Market/Financial: {mkt_done}/{num_tech}")
    st.sidebar.write(f"Capability: {cap_done}/{num_tech}")
    st.sidebar.write(f"ESG: {env_done}/{num_tech}")
    st.sidebar.write(f"Life Cycle: {lc_done}/{num_tech}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **About This Tool**
    
    Educational application for multi-criteria
    technology assessment and portfolio ranking.
    
    Designed for university courses on technology
    strategy and technology assessment.
    """)


if __name__ == "__main__":
    main()

