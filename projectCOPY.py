import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter


def extract_numeric_vals(df: pd.DataFrame) -> np.ndarray: #Tool for csv extraction
    numeric_vals = df.select_dtypes(include=[np.number])
    if numeric_vals.empty:
        raise ValueError("No numeric columns found in the CSV.")
    return numeric_vals.to_numpy().flatten()


def reset_all_sliders():
    for i in range(total_params):
        slider_key = f"slider_{name}_{i}"
        st.session_state[slider_key] = float(params[i])


def most_common(data):
    if data is None:
        return None
    # ensure a flat list
    arr = list(np.asarray(data).ravel())
    if not arr:
        return None
    counts = Counter(arr)
    # return the single value with highest count (ties -> arbitrary one)
    return max(counts.keys(), key=lambda k: counts[k])

            


# Page Config + Styling
st.set_page_config(
    page_title="Histogram Fitting App",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>
.block-container {
    max-width: 1100px;
    padding-left: 3.5rem;
    padding-right: 3.5rem;
}
</style>
""", unsafe_allow_html=True)


st.title("📊 Histogram Fitting Web App")
st.write("By Lucas Kovacevic")
st.markdown("---")
sidebar = st.sidebar
# -----------------------------------------------------------



# Initialize all session_state variables that will be used
if 'data' not in st.session_state: #our data needs a global state
    st.session_state.data = None

if 'fit_params' not in st.session_state: #our fit parameters will need to be acessed by more than one object
    st.session_state.fit_params = None

# -----------------------------------------------------------

# Columns Layout
col1, col2 = st.columns([0.3, 0.7])
# -----------------------------------------------------------

# Left Column: Data Input
with col1:

    st.subheader("📝 Enter Your Data")

    raw_data = st.text_area(
        "Paste numeric data:",
        height=100
    )

    parse_clicked = st.button("Confirm Data")



    uploaded_csv = st.file_uploader(
        "Upload CSV containing numeric data",
        type=["csv"]
    )

    bins = st.slider(
    '# Of Histogram Bins',
    min_value=5,
    max_value=300,
    value=30,
    key='bins'
    )

    # If CSV uploaded → load it
    if uploaded_csv is not None:
        try:
            df = pd.read_csv(uploaded_csv)
            numeric_vals = extract_numeric_vals(df)
            st.session_state.data = numeric_vals #update session state of data
        except Exception as e:
            st.error(f"❌ Error processing CSV: {e}")

    show_fit = st.toggle("Show fitted distribution(s)")
# -----------------------------------------------------------


# Handling of the parse_clicked button
if parse_clicked:
    try:
        data = [
            float(x)
            for x in raw_data.replace("\n", " ").replace(",", " ").split() #simple parse
            if x.strip() != ""
        ]
        st.session_state.data = data #update session state data
    except:
        st.error("Could not parse the data ― ensure values are numeric.")


# -----------------------------------------------------------


# convert data to nparray
# -----------------------------------------------------------
if st.session_state.data is not None:
    data_np = np.asarray(st.session_state.data, dtype=float).flatten()

# -----------------------------------------------------------
# Distribution selector
# -----------------------------------------------------------
dists = {
    'Normal (Gaussian)': stats.norm,
    'Gamma': stats.gamma,
    'Weibull (min)': stats.weibull_min,
    'Log-Normal': stats.lognorm,
    'Exponential': stats.expon,
    'Beta': stats.beta,
    'Chi-squared': stats.chi2,
    'Student-t': stats.t,
    'Uniform': stats.uniform,
    'Laplace': stats.laplace
}

st.header("Choose Distribution to Fit")

dist_name_select = st.multiselect("Choose Distribution to Fit", list(dists.keys()))


# -----------------------------------------------------------


# handle fitting parameters here
# -----------------------------------------------------------
if st.session_state.data is not None:
        if not dist_name_select:
            st.session_state.fit_params ={}
        else:


            st.session_state.fit_params = {}
            for i in dist_name_select:
                selected_dist = dists[i]
                try:
                    params = selected_dist.fit(data_np)
                    st.session_state.fit_params[i] = params #add the params to the dictionary, with the key being the corresponding distribution
                except Exception as e:
                    st.warning(f"Failed to fit {i}: {e}")

        # Update stored distribution name
# -----------------------------------------------------------


# Right Column: Plotting
# -----------------------------------------------------------
with col2:
    # Create the header for the right-hand column
    st.header("📈 Histogram + Model Fitting")
    # Create a matplotlib figure and axis (do NOT force a dark facecolor here)
    fig, ax = plt.subplots(figsize=(8, 5))
    # fig, ax = plt.subplots(...) creates a new Figure and Axes for plotting.
    # We will make the figure/axes background transparent so the app's own background shows through.

    # Make figure and axes backgrounds transparent so the app page background is visible.
    fig.patch.set_alpha(0.0)
    # fig.patch.set_alpha(0.0) makes the figure canvas transparent.
    ax.patch.set_alpha(0.0)
    # ax.patch.set_alpha(0.0) makes the plotting area (axes) transparent.

    # Always set axis labels/tick colors to white so they are visible on dark themes.
    # (You requested white even when no data exists; this forces that color.)
    ax.set_xlabel("Value", color="white")
    # Set X-axis label and force its color to white.
    ax.set_ylabel("Density", color="white")
    # Set Y-axis label and force its color to white.

    ax.tick_params(axis="x", colors="white")
    # Set X-axis tick label color to white (ensures numbers are white).
    ax.tick_params(axis="y", colors="white")
    # Set Y-axis tick label color to white.
    if dist_name_select == None:
            st.session_state.fit_params = None
    # Only proceed if we have data to plot
    if st.session_state.data is not None:
        # Convert session data to a flat nump y array (already done earlier, but ensure local var exists)
        data_np = np.asarray(st.session_state.data, dtype=float).flatten()
        n, bins, patches = ax.hist(
            data_np,                        # the flattened numeric array to plot
            bins=st.session_state.bins,     # number of bins controlled by the slider stored in session_state
            density=True,                   # normalize histogram to match PDF scale
            alpha=1.0,                      # bar opacity (1.0 = opaque)
            color="teal",              # bar fill color (visible on dark backgrounds)
            edgecolor="cyan"               # bar edge color for contrast
        )
        hist_max = n.max()

        ax.set_ylim([0, hist_max*1.05])
        # Prepare a list of colors to cycle through for fitted distribution lines
        colors = ['red', 'pink', 'lime', 'orange', 'magenta', 'yellow', 'deepskyblue', 'gold', 'violet', 'springgreen']
        # If user requested fitted curves and we have stored fit parameters, draw them
        if show_fit and st.session_state.fit_params:
            x = np.linspace(min(data_np), max(data_np), 500)
            colors = ['red','pink','lime','orange','magenta','yellow','deepskyblue','gold','violet','springgreen']
            for idx, name in enumerate(list(st.session_state.fit_params.keys())):
                params = st.session_state.fit_params.get(name)
                if params is None:
                    continue
                dist_obj = dists.get(name)
                
                # Collect parameter values from sliders (or use fitted if no sliders yet)
                use_params = []
                num_shapes = getattr(dist_obj, "numargs", 0) or 0
                total_params = num_shapes + 2
                for i in range(total_params):
                    slider_key = f"slider_{name}_{i}"
                    use_params.append(st.session_state.get(slider_key, params[i]))
                use_params = tuple(use_params)
                
                try:
                    pdf_y = dist_obj.pdf(x, *use_params)
                except Exception:
                    pdf_y = dist_obj.pdf(x, *(use_params[-2:]))
                ax.plot(x, pdf_y, lw=2, color=colors[idx % len(colors)], label=name)
    # - loc='upper center' anchors the legend horizontally centered relative to bbox_to_anchor
        # - bbox_to_anchor=(0.5, -0.18) positions the legend below the axes (0.5 = center)
        # - ncol controls how many columns the legend has (auto-compute a reasonable number)
        ncols = max(1, min(4, len(st.session_state.fit_params) if st.session_state.fit_params else 1))
        legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=ncols, framealpha=0.9, facecolor="#222222", edgecolor="white")
        if legend:
            for t in legend.get_texts():
                t.set_color("white")
        fig.subplots_adjust(bottom=0.38)  # leave more room for the control box below

    else:
        # No data: show an informative placeholder message inside the plot area
        ax.text(0.5, 0.5, "No data loaded\nPaste or upload CSV and click 'Confirm Data'",
                ha='center', va='center', color='white', fontsize=12, transform=ax.transAxes)
        # ax.text(...) places centered white text in the axes to tell the user what to do.


    # Render the final figure in Streamlit
    st.pyplot(fig)
    # st.pyplot(fig) displays the Matplotlib figure in the Streamlit app.

    # ...existing code...
    with st.expander("Fit details (per selected distribution)", expanded=False):
        if not dist_name_select:
            st.write("No distributions selected.")
        else:
            fit_params = st.session_state.get("fit_params") or {}
            param_overrides = st.session_state.get("param_overrides") or {}
            for name in dist_name_select:
                with st.container():
                    st.markdown(f"### {name}")


                params = fit_params.get(name)
                if params is None:
                    st.write("No fitted parameters available.")
                else:
                    st.write("Fitted parameters:", params)

                    # --- Use slider values instead of fitted params ---
                    num_shapes = getattr(dists[name], "numargs", 0) or 0
                    total_params = num_shapes + 2
                    use_params = []
                    for i in range(total_params):
                        slider_key = f"slider_{name}_{i}"
                        use_params.append(st.session_state.get(slider_key, params[i]))
                    use_params = tuple(use_params)

                    # Compute quality-of-fit metrics with current slider values
                    dist_obj = dists[name]
                    observed, bin_edges = np.histogram(data_np, bins=st.session_state.bins, density=False)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
                    expected = dist_obj.pdf(bin_centers, *use_params) * len(data_np) * (bin_edges[1] - bin_edges[0])
                    avg_error = np.mean(np.abs(expected - observed))
                    expected = np.maximum(expected, 1e-10)
                    chi2_stat = np.sum((observed - expected) ** 2 / expected)

                    logpdf_vals = dist_obj.logpdf(data_np, *use_params)
                    logpdf_vals = logpdf_vals[np.isfinite(logpdf_vals)]
                    nll = -np.sum(logpdf_vals)

                    k = len(use_params)
                    n = len(data_np)
                    aic = 2 * k + 2 * nll
                    bic = k * np.log(n) + 2 * nll

                    st.write("### Goodness of Fit (Quality Metrics)")
                    st.write(f"- **Negative Log-Likelihood (NLL):** {nll:.4f}")
                    st.write(f"- **AIC:** {aic:.4f}  *(lower = better)*")
                    st.write(f"- **BIC:** {bic:.4f}  *(lower = better)*")
                    st.write(f"- **Avg Absolute Error Between Curve and Histogram:** {avg_error:.4f} units")
                    st.write(f"- **Chi-squared**: {chi2_stat:.4f} (lower = better)")


                    if np.any(expected < 5):
                        st.caption("⚠️ Some expected bin counts < 5 — Chi² approximation may be inaccurate.")

                        dist_obj = dists[name]

                        # Choose appropriate number of bins
                        num_bins = min(30, max(5, int(len(data_np) ** 0.5)))

                        observed, bin_edges = np.histogram(data_np, bins=num_bins, density=False)

                        # Expected count for each bin: N * (CDF(b_{i+1}) - CDF(b_i))
                        expected = np.array([
                            len(data_np) * (
                                dist_obj.cdf(bin_edges[i+1], *params) -
                                dist_obj.cdf(bin_edges[i], *params)
                            )
                            for i in range(num_bins)
                        ])

                        # Prevent zero-division
                        expected = np.maximum(expected, 1e-12)

                        chi2_stat = np.sum((observed - expected)**2 / expected)

                        


                        # -----------------------------
                        ## -----------------------------
                        # 2. Anderson–Darling Test (SciPy supports VERY few dists)
                        # -----------------------------


                    num_shapes = getattr(dist_obj, "numargs", 0) or 0
                    total_params = num_shapes + 2
                    
                    # Initialize overrides dict if needed
                    if "param_overrides" not in st.session_state:
                        st.session_state.param_overrides = {}
                    if name not in st.session_state.param_overrides:
                        st.session_state.param_overrides[name] = list(params)
                    
                    overrides = st.session_state.param_overrides[name]
                    
                                        # Create one slider per parameter
                    for i in range(total_params):
                        if i < num_shapes:
                            label = f"shape {i+1}"
                        elif i == num_shapes:
                            label = "loc"
                        else:
                            label = "scale"
                        
                        # Compute slider bounds
                        bounds_key = f"slider_bounds_{name}_{i}"
                        if bounds_key not in st.session_state:
                            if params[i] == 0:
                                lo, hi = -10.0, 10.0
                            else:
                                span = max(abs(params[i]) * 5.0, 1.0)
                                lo, hi = params[i] - span, params[i] + span
                            st.session_state[bounds_key] = (float(lo), float(hi))
                        
                        lo, hi = st.session_state[bounds_key]
                        
                        # Use session_state key directly for slider so it persists immediately
                        slider_key = f"slider_{name}_{i}"
                        if slider_key not in st.session_state:
                            st.session_state[slider_key] = float(params[i])
                        
                        val = st.slider(
                            f"{label}",
                            min_value=float(lo),
                            max_value=float(hi),
                            value=st.session_state[slider_key],
                            key=slider_key
                        )
                    
                    # Return to default fit button
                                       # Return to default fit button
                    def make_reset_callback(dist_name, dist_params):
                        def reset_callback():
                            dist_obj = dists[dist_name]
                            num_shapes = getattr(dist_obj, "numargs", 0) or 0
                            total_p = num_shapes + 2
                            for i in range(total_p):
                                slider_key = f"slider_{dist_name}_{i}"
                                st.session_state[slider_key] = float(dist_params[i])
                        return reset_callback
                    

                    st.button("Return to default fit", key=f"reset_{name}", on_click=make_reset_callback(name, params))
