import base64
import io

import joblib
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px

# -------------------------
# Load Trained Model
# -------------------------
# NOTE: Ensure 'model.pkl' is available in the same directory as this app.py file
try:
    model = joblib.load("model.pkl")
except FileNotFoundError:
    print("Error: 'model.pkl' not found. Please ensure the trained model file is present.")
    model = None


# -------------------------
# Preprocessing
# -------------------------
def preprocess(df: pd.DataFrame):
    """Preprocess uploaded CSV to align with the training pipeline.

    Supports:
      - Raw step-level logs (Battery_ID, Cycle_Index, Discharge_Capacity(Ah), etc.)
      - Cycle-level aggregates (Discharge_Capacity(Ah)_max, etc.)
      - Already engineered snake_case / rolling-feature datasets.

    Returns:
      df_features: DataFrame with columns == model.feature_names_in_
      plot_df:    DataFrame with cycle_index, capacity, (and battery_id if present) for plotting
    """
    if model is None or df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = df.copy()
    # Drop obvious junk columns
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

    # ------------------------
    # Helper: build cycle-level features from raw step data
    # ------------------------
    def build_cycle_features_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
        required = {"Battery_ID", "Cycle_Index"}
        if not required.issubset(raw.columns):
            return pd.DataFrame()

        group_cols = ["Battery_ID", "Cycle_Index"]
        if "Protocol_ID" in raw.columns:
            group_cols.append("Protocol_ID")

        agg = {}

        if "Discharge_Capacity(Ah)" in raw.columns:
            agg["Discharge_Capacity(Ah)"] = ["max", "mean", "min"]
        if "Charge_Capacity(Ah)" in raw.columns:
            agg["Charge_Capacity(Ah)"] = ["max", "mean", "min"]
        if "Voltage(V)" in raw.columns:
            agg["Voltage(V)"] = ["max", "mean", "min", "std"]
        if "Current(A)" in raw.columns:
            agg["Current(A)"] = ["mean", "std"]
        if "Discharge_Energy(Wh)" in raw.columns:
            agg["Discharge_Energy(Wh)"] = ["max"]
        if "Charge_Energy(Wh)" in raw.columns:
            agg["Charge_Energy(Wh)"] = ["max"]
        if "Aux_Temperature_1(C)" in raw.columns:
            agg["Aux_Temperature_1(C)"] = ["mean", "max", "min", "std"]

        if not agg:
            return pd.DataFrame()

        cycle_stats = raw.groupby(group_cols).agg(agg).reset_index()

        # Flatten MultiIndex column names
        cycle_stats.columns = [
            "_".join([str(c) for c in col if c]).rstrip("_")
            for col in cycle_stats.columns.to_flat_index()
        ]

        # |Current| mean per cycle
        if "Current(A)" in raw.columns:
            abs_mean = (
                raw.assign(Current_abs=raw["Current(A)"].abs())
                .groupby(["Battery_ID", "Cycle_Index"])["Current_abs"]
                .mean()
                .reset_index(name="Current(A)_abs_mean")
            )
            cycle_stats = cycle_stats.merge(
                abs_mean, on=["Battery_ID", "Cycle_Index"], how="left"
            )

        # SOH(%) from first-cycle max discharge capacity per battery
        if "Discharge_Capacity(Ah)_max" in cycle_stats.columns:
            nominal_caps = {}
            for bid, sub in cycle_stats.groupby("Battery_ID"):
                first_idx = sub["Cycle_Index"].min()
                first_row = sub[sub["Cycle_Index"] == first_idx]
                if not first_row.empty:
                    nominal_caps[bid] = float(
                        first_row["Discharge_Capacity(Ah)_max"].iloc[0]
                    )

            def _soh(row):
                nom = nominal_caps.get(row["Battery_ID"])
                if nom and nom > 0:
                    return 100.0 * row["Discharge_Capacity(Ah)_max"] / nom
                return np.nan

            cycle_stats["SOH(%)"] = cycle_stats.apply(_soh, axis=1)

        return cycle_stats

    # ------------------------
    # Detect input style
    # ------------------------
    has_final_like = (
        "soh_percent" in df.columns
        and any(col.startswith("rolling_") for col in df.columns)
    )
    has_cycle_agg = (
        "Discharge_Capacity(Ah)_max" in df.columns
        or "discharge_capacity_ah_max" in df.columns
    )
    has_raw = {
        "Discharge_Capacity(Ah)",
        "Cycle_Index",
        "Battery_ID",
    }.issubset(df.columns)

    if has_final_like:
        cycle_df = df.copy()
    elif has_cycle_agg and not has_raw:
        cycle_df = df.copy()
    elif has_raw:
        cycle_df = build_cycle_features_from_raw(df)
    else:
        print(
            "Unsupported input format. Expect raw step data, cycle aggregates, or engineered final_df-style data."
        )
        return pd.DataFrame(), pd.DataFrame()

    if cycle_df is None or cycle_df.empty:
        print("Cycle-level feature construction failed.")
        return pd.DataFrame(), pd.DataFrame()

    # ------------------------
    # Rename to snake_case (to match feature_engineering pipeline)
    # ------------------------
    rename_mapping = {
        "Battery_ID": "battery_id",
        "Cycle_Index": "cycle_index",
        "Protocol_ID": "protocol_id",
        "Discharge_Capacity(Ah)_max": "discharge_capacity_ah_max",
        "Discharge_Capacity(Ah)_mean": "discharge_capacity_ah_mean",
        "Discharge_Capacity(Ah)_min": "discharge_capacity_ah_min",
        "Charge_Capacity(Ah)_max": "charge_capacity_ah_max",
        "Charge_Capacity(Ah)_mean": "charge_capacity_ah_mean",
        "Charge_Capacity(Ah)_min": "charge_capacity_ah_min",
        "Voltage(V)_max": "voltage_v_max",
        "Voltage(V)_mean": "voltage_v_mean",
        "Voltage(V)_min": "voltage_v_min",
        "Voltage(V)_std": "voltage_v_std",
        "Current(A)_mean": "current_a_mean",
        "Current(A)_std": "current_a_std",
        "Discharge_Energy(Wh)_max": "discharge_energy_wh_max",
        "Charge_Energy(Wh)_max": "charge_energy_wh_max",
        "Aux_Temperature_1(C)_mean": "aux_temperature_1_c_mean",
        "Aux_Temperature_1(C)_max": "aux_temperature_1_c_max",
        "Aux_Temperature_1(C)_min": "aux_temperature_1_c_min",
        "Aux_Temperature_1(C)_std": "aux_temperature_1_c_std",
        "Current(A)_abs_mean": "current_a_abs_mean",
        "SOH(%)": "soh_percent",
        "RUL": "rul",
    }
    cols_to_rename = {k: v for k, v in rename_mapping.items() if k in cycle_df.columns}
    cycle_df = cycle_df.rename(columns=cols_to_rename)

    # Ensure key identifiers exist
    if "battery_id" not in cycle_df.columns and "Battery_ID" in cycle_df.columns:
        cycle_df = cycle_df.rename(columns={"Battery_ID": "battery_id"})
    if "cycle_index" not in cycle_df.columns and "Cycle_Index" in cycle_df.columns:
        cycle_df = cycle_df.rename(columns={"Cycle_Index": "cycle_index"})

    # If still no battery_id, assume single battery
    if "battery_id" not in cycle_df.columns:
        cycle_df["battery_id"] = 0

    # If no cycle_index, create per-battery running index
    if "cycle_index" not in cycle_df.columns:
        cycle_df["cycle_index"] = cycle_df.groupby("battery_id").cumcount() + 1

    # Sort for consistent rolling & plotting
    cycle_df = cycle_df.sort_values(
        ["battery_id", "cycle_index"]
    ).reset_index(drop=True)

    # ------------------------
    # Derive helper columns
    # ------------------------
    if (
        "discharge_capacity_ah_max" in cycle_df.columns
        and "capacity" not in cycle_df.columns
    ):
        cycle_df["capacity"] = cycle_df["discharge_capacity_ah_max"]

    # Compute SOH if missing but discharge_capacity_ah_max present
    if "soh_percent" not in cycle_df.columns and "discharge_capacity_ah_max" in cycle_df.columns:
        nominal_caps = {}
        for bid, sub in cycle_df.groupby("battery_id"):
            first = sub["discharge_capacity_ah_max"].iloc[0]
            if pd.notna(first) and first > 0:
                nominal_caps[bid] = float(first)

        def _soh2(row):
            nom = nominal_caps.get(row["battery_id"])
            cap = row.get("discharge_capacity_ah_max", np.nan)
            if nom and pd.notna(cap) and nom > 0:
                return 100.0 * cap / nom
            return np.nan

        cycle_df["soh_percent"] = cycle_df.apply(_soh2, axis=1)

    # ------------------------
    # Rolling features (window = 5 cycles)
    # ------------------------
    rolling_base = [
        "discharge_capacity_ah_max",
        "charge_capacity_ah_max",
        "voltage_v_max",
        "current_a_mean",
        "discharge_energy_wh_max",
        "charge_energy_wh_max",
        "aux_temperature_1_c_mean",
        "current_a_abs_mean",
        "soh_percent",
    ]

    for feature in rolling_base:
        if feature not in cycle_df.columns:
            continue

        cycle_df[f"rolling_mean_{feature}"] = (
            cycle_df.groupby("battery_id")[feature]
            .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        )
        cycle_df[f"rolling_std_{feature}"] = (
            cycle_df.groupby("battery_id")[feature]
            .transform(lambda x: x.rolling(window=5, min_periods=1).std().fillna(0.0))
        )

    # ------------------------
    # Align with model's expected features
    # ------------------------
    if not hasattr(model, "feature_names_in_"):
        print("Model does not expose feature_names_in_. Cannot align features.")
        return pd.DataFrame(), pd.DataFrame()

    feature_cols = list(model.feature_names_in_)
    missing_for_model = [c for c in feature_cols if c not in cycle_df.columns]
    if missing_for_model:
        print(f"Error: Missing final feature columns for prediction: {missing_for_model}")
        return pd.DataFrame(), pd.DataFrame()

    df_features = cycle_df[feature_cols].copy()
    df_features = df_features.dropna(subset=feature_cols)

    if df_features.empty:
        print("All rows dropped after aligning to model features.")
        return pd.DataFrame(), pd.DataFrame()

    # Plotting dataframe aligned to rows kept for prediction
    plot_df = cycle_df.loc[df_features.index].copy()

    if "cycle_index" not in plot_df.columns:
        plot_df["cycle_index"] = plot_df.groupby("battery_id").cumcount() + 1

    if "capacity" not in plot_df.columns:
        if "discharge_capacity_ah_max" in plot_df.columns:
            plot_df["capacity"] = plot_df["discharge_capacity_ah_max"]

    return df_features, plot_df


# -------------------------
# Dash App Layout
# -------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        html.H2(
            "Battery Remaining Useful Life (RUL) Predictor",
            className="text-center mt-4 mb-3",
        ),
        html.P(
            "Upload a CSV in the same format as your raw experiment data or engineered cycle data.",
            className="text-center text-muted mb-4",
        ),
        dcc.Upload(
            id="upload-data",
            children=html.Div(
                ["Drag and Drop or ", html.A("Select CSV File", className="text-primary")],
            ),
            style={
                "width": "100%",
                "height": "70px",
                "lineHeight": "70px",
                "borderWidth": "2px",
                "borderStyle": "dashed",
                "borderRadius": "8px",
                "textAlign": "center",
                "margin": "15px 0",
                "backgroundColor": "#f9f9f9",
                "cursor": "pointer",
            },
            multiple=False,
        ),
        html.Div(
            id="status-msg",
            children="Upload a CSV file containing battery cycle data to begin.",
            className="text-center text-muted mb-4",
        ),
        dbc.Card(
            dbc.CardBody(
                [
                    html.H3(
                        "RUL Prediction for Latest Cycle:",
                        className="card-title text-center",
                    ),
                    html.Div(
                        id="prediction-output",
                        children="--",
                        className="text-center text-primary",
                        style={"fontSize": "32px", "fontWeight": "bold"},
                    ),
                ]
            ),
            className="mb-4 shadow-sm",
        ),
        dcc.Graph(id="degradation-plot", style={"height": "500px"}),
    ],
    fluid=True,
)


# -------------------------
# Callback
# -------------------------
@app.callback(
    Output("prediction-output", "children"),
    Output("status-msg", "children"),
    Output("degradation-plot", "figure"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
def update_output(contents, filename):
    if contents is None:
        return (
            "--",
            "Upload a CSV file containing battery cycle data to begin.",
            {},
        )

    if model is None:
        return (
            "--",
            "Model file 'model.pkl' not found. Please add it next to app.py and restart.",
            {},
        )

    try:
        # Decode uploaded content
        content_type, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)

        # Try UTF-8 then fall back
        try:
            s = decoded.decode("utf-8")
        except UnicodeDecodeError:
            s = decoded.decode("ISO-8859-1")

        df_raw = pd.read_csv(io.StringIO(s))

        if df_raw is None or df_raw.empty:
            return "--", "Uploaded file is empty or unreadable.", {}

        df_features, plot_df = preprocess(df_raw)

        if df_features.empty or plot_df.empty:
            return (
                "--",
                "Preprocessing failed. Ensure the CSV has either raw step data or cycle-level aggregates similar to the training data.",
                {},
            )

        # Predict RUL for latest available cycle
        latest = df_features.tail(1)
        y_pred = model.predict(latest)

        if isinstance(y_pred, (list, np.ndarray)):
            prediction = float(y_pred[0])
        else:
            prediction = float(y_pred)

        prediction = max(0.0, prediction)

        # -------------------------
        # Build degradation plot
        # -------------------------
        fig = {}

        if "cycle_index" in plot_df.columns and (
            "capacity" in plot_df.columns
            or "discharge_capacity_ah_max" in plot_df.columns
        ):
            y_col = (
                "capacity"
                if "capacity" in plot_df.columns
                else "discharge_capacity_ah_max"
            )

            if "battery_id" in plot_df.columns:
                fig = px.line(
                    plot_df,
                    x="cycle_index",
                    y=y_col,
                    color="battery_id",
                    title="Capacity Degradation Over Cycles",
                )
            else:
                fig = px.line(
                    plot_df,
                    x="cycle_index",
                    y=y_col,
                    title="Capacity Degradation Over Cycles",
                )

            # EOL line at 80% of initial capacity
            try:
                first_cap = (
                    plot_df.sort_values(
                        ["battery_id", "cycle_index"]
                        if "battery_id" in plot_df.columns
                        else ["cycle_index"]
                    )[y_col]
                    .iloc[0]
                )
                if pd.notna(first_cap) and first_cap > 0:
                    eol_level = 0.8 * first_cap
                    fig.add_hline(
                        y=eol_level,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="80% EOL",
                    )
            except Exception:
                pass

            fig.update_layout(
                xaxis_title="Cycle Index",
                yaxis_title="Capacity (Ah)",
                hovermode="x unified",
            )

        status = (
            f"Successfully analyzed file: {filename}. "
            f"Predicted RUL for the latest cycle is {prediction:.2f} cycles."
        )

        return f"{prediction:.2f} cycles", status, fig

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return (
            "--",
            f"Error processing file: {filename}. Please check file format and required columns. Details: {e}",
            {},
        )


if __name__ == "__main__":
    app.run_server(debug=True)
