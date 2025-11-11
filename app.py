import base64
import io
import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, callback_context
from dash.dash_table import DataTable
import dash_bootstrap_components as dbc
from model_inference_example import BatteryRULPredictor  # Import the class

# Initialize the Dash app with a modern theme
app = Dash(__name__, external_stylesheets=[
    dbc.themes.LUX,
    'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@400;500;600;700;800&display=swap'
], suppress_callback_exceptions=True)
app.title = "Battery RUL Predictor"

# Initialize the predictors (both models)
predictor_zenodo = BatteryRULPredictor(model_type='zenodo')
predictor_nasa = BatteryRULPredictor(model_type='nasa')

# Protocol descriptions
PROTOCOL_DESCRIPTIONS = {
    1: "Regular charging and current - Standard usage pattern",
    2: "Normal use - Regular driving, avoids complete battery drain",
    3: "Deep use / Full charge habit - Drains to near empty, charges to 100%",
    4: "Fast charge - Frequent fast charging with normal driving",
    5: "Very fast charge - High-speed charging almost every time",
    6: "Super-fast charge - Highest charging rate consistently",
    7: "Hard driving - Aggressive acceleration or heavy load, normal charging",
    8: "Heavy load - Uphill driving, cargo, or towing",
    9: "Fast + Deep - Fast charging with deep discharges",
    10: "Fast + Full - Quick charging always to full capacity",
    11: "Very fast + Deep - Very fast charging with deep discharges",
    12: "Very fast + Full - Full charge at very high speed",
    13: "Hard + Fast - Heavy driving with very fast charging",
    14: "Light use - Gentle driving, light load, rare hard acceleration",
    15: "Moderate use - Balanced daily driving and usage",
    16: "Stop & Go - City driving with frequent acceleration/braking"
}

# Detailed protocol information for the info modal
PROTOCOL_DETAILS = {
    1: {"title": "Protocol 1: Regular Charging", "description": "Regular charging and current. Standard usage pattern with typical charging behavior."},
    2: {"title": "Protocol 2: Normal Use", "description": "Normal use driving, avoids letting the battery drain completely. Best for daily commuting."},
    3: {"title": "Protocol 3: Deep Use / Full Charge", "description": "Often drives until the battery is almost empty before charging and always charges to 100%. Can reduce battery lifespan."},
    4: {"title": "Protocol 4: Fast Charge", "description": "Frequently uses fast chargers, but drives normally. Convenient for busy schedules."},
    5: {"title": "Protocol 5: Very Fast Charge", "description": "Prefers high-speed charging almost every time. Saves time but may impact battery health."},
    6: {"title": "Protocol 6: Super-Fast Charge", "description": "Uses the highest charging rate consistently. Maximum convenience with potential battery stress."},
    7: {"title": "Protocol 7: Hard Driving", "description": "Drives aggressively, accelerates hard, or carries a heavy load but charges normally. High power demand."},
    8: {"title": "Protocol 8: Heavy Load", "description": "Drives with a heavy load, such as uphill, with cargo, or towing. Consistent high-stress usage."},
    9: {"title": "Protocol 9: Fast + Deep", "description": "Uses fast charging and often lets the battery drain to a very low level. Combines two stressful patterns."},
    10: {"title": "Protocol 10: Fast + Full", "description": "Charges quickly and always charges to full capacity. Convenient but potentially stressful."},
    11: {"title": "Protocol 11: Very Fast + Deep", "description": "Combines very fast charging with frequent deep discharges. High-stress combination."},
    12: {"title": "Protocol 12: Very Fast + Full", "description": "Always charges to full at a very high charging speed. Maximum stress pattern."},
    13: {"title": "Protocol 13: Hard + Fast", "description": "Heavy driving combined with very fast charging. Intense usage pattern."},
    14: {"title": "Protocol 14: Light Use", "description": "Gentle driving with a light load, rarely accelerates hard. Optimal for battery longevity."},
    15: {"title": "Protocol 15: Moderate Use", "description": "Comfortable daily driving and balanced usage. Well-balanced approach."},
    16: {"title": "Protocol 16: Stop & Go", "description": "City-style driving with frequent acceleration and braking, similar to traffic patterns. Typical urban usage."}
}

# Custom CSS styles
CUSTOM_STYLE = {
    'hero-section': {
        'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'color': 'white',
        'padding': '80px 30px',
        'borderRadius': '20px',
        'marginBottom': '40px',
        'boxShadow': '0 20px 60px rgba(102, 126, 234, 0.3)',
        'fontFamily': 'Poppins, sans-serif'
    },
    'card-style': {
        'borderRadius': '20px',
        'boxShadow': '0 8px 16px rgba(0,0,0,0.08)',
        'marginBottom': '25px',
        'border': 'none',
        'transition': 'transform 0.3s ease, box-shadow 0.3s ease',
        'fontFamily': 'Inter, sans-serif'
    },
    'button-primary': {
        'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'border': 'none',
        'borderRadius': '30px',
        'padding': '15px 45px',
        'fontSize': '18px',
        'fontWeight': '600',
        'boxShadow': '0 6px 20px rgba(102, 126, 234, 0.4)',
        'transition': 'all 0.3s ease',
        'fontFamily': 'Poppins, sans-serif',
        'letterSpacing': '0.5px',
        'textTransform': 'none'
    },
    'feature-card': {
        'backgroundColor': 'white',
        'borderRadius': '20px',
        'padding': '40px 30px',
        'textAlign': 'center',
        'boxShadow': '0 8px 16px rgba(0,0,0,0.08)',
        'height': '100%',
        'transition': 'transform 0.3s ease, box-shadow 0.3s ease',
        'fontFamily': 'Inter, sans-serif',
        'border': '1px solid rgba(102, 126, 234, 0.1)'
    },
    'section-title': {
        'fontFamily': 'Poppins, sans-serif',
        'fontWeight': '700',
        'color': '#2d3748',
        'marginBottom': '30px',
        'fontSize': '36px',
        'letterSpacing': '-0.5px'
    },
    'body-text': {
        'fontFamily': 'Inter, sans-serif',
        'fontSize': '17px',
        'lineHeight': '1.8',
        'color': '#4a5568',
        'fontWeight': '400'
    }
}

# Landing Page Layout
def landing_page():
    return dbc.Container([
        # Hero Section
        html.Div([
            html.H1("🔋 Battery RUL Prediction System", 
                   className="display-2 text-center mb-4",
                   style={
                       'textShadow': '2px 2px 8px rgba(0,0,0,0.2)',
                       'fontFamily': 'Poppins, sans-serif',
                       'fontWeight': '800',
                       'letterSpacing': '-1px'
                   }),
            html.P("Advanced Machine Learning for Battery Lifetime Prediction",
                  className="text-center mb-5",
                  style={
                      'fontSize': '22px',
                      'fontFamily': 'Inter, sans-serif',
                      'fontWeight': '300',
                      'opacity': '0.95',
                      'letterSpacing': '0.3px'
                  }),
                    html.Div([
                        dcc.Link(
                            dbc.Button("Start Zenodo Prediction →", 
                                      color="light",
                                      size="lg",
                                      className="mt-3 me-3",
                                      style=CUSTOM_STYLE['button-primary']),
                            href='/prediction',
                            style={'textDecoration': 'none'}
                        ),
                        dcc.Link(
                            dbc.Button("Start NASA Prediction →", 
                                      color="light",
                                      size="lg",
                                      className="mt-3",
                                      style={**CUSTOM_STYLE['button-primary'], 'background': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)'}),
                            href='/nasa-prediction',
                            style={'textDecoration': 'none'}
                        )
                    ], className="text-center")
        ], style=CUSTOM_STYLE['hero-section']),

        # Project Description
        dbc.Row([
            dbc.Col([
                html.H2("About This Project", 
                       className="text-center mb-4",
                       style=CUSTOM_STYLE['section-title']),
                html.P([
                    "This intelligent system predicts the ",
                    html.Strong("Remaining Useful Life (RUL)", style={'color': '#667eea'}),
                    " of lithium-ion batteries using state-of-the-art machine learning models. "
                    "Our solution helps optimize battery management, reduce costs, and improve sustainability."
                ], className="text-center mb-5", 
                   style={**CUSTOM_STYLE['body-text'], 'maxWidth': '800px', 'margin': '0 auto'})
            ], width=12)
        ]),

        # Features Section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div("📊", style={'fontSize': '56px', 'marginBottom': '20px'}),
                        html.H4("Data-Driven", 
                               className="mb-3",
                               style={'fontFamily': 'Poppins, sans-serif', 'fontWeight': '700', 'color': '#2d3748'}),
                        html.P("Trained on extensive battery cycle data from NASA and Zenodo datasets",
                              style={'fontSize': '15px', 'color': '#718096', 'lineHeight': '1.7', 'fontFamily': 'Inter, sans-serif'})
                    ])
                ], style=CUSTOM_STYLE['feature-card'], className="hover-lift")
            ], md=4, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div("🤖", style={'fontSize': '56px', 'marginBottom': '20px'}),
                        html.H4("AI-Powered", 
                               className="mb-3",
                               style={'fontFamily': 'Poppins, sans-serif', 'fontWeight': '700', 'color': '#2d3748'}),
                        html.P("Advanced machine learning algorithms for accurate RUL prediction",
                              style={'fontSize': '15px', 'color': '#718096', 'lineHeight': '1.7', 'fontFamily': 'Inter, sans-serif'})
                    ])
                ], style=CUSTOM_STYLE['feature-card'], className="hover-lift")
            ], md=4, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div("⚡", style={'fontSize': '56px', 'marginBottom': '20px'}),
                        html.H4("Fast & Reliable", 
                               className="mb-3",
                               style={'fontFamily': 'Poppins, sans-serif', 'fontWeight': '700', 'color': '#2d3748'}),
                        html.P("Instant predictions across 16 different battery protocols",
                              style={'fontSize': '15px', 'color': '#718096', 'lineHeight': '1.7', 'fontFamily': 'Inter, sans-serif'})
                    ])
                ], style=CUSTOM_STYLE['feature-card'], className="hover-lift")
            ], md=4, className="mb-4"),
        ], className="mb-5"),

        # How It Works Section
        dbc.Row([
            dbc.Col([
                html.H2("How It Works", 
                       className="text-center mb-5",
                       style=CUSTOM_STYLE['section-title']),
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H1("1", className="text-center mb-3",
                               style={'color': '#667eea', 'fontSize': '52px', 'fontWeight': '800', 'fontFamily': 'Poppins, sans-serif'}),
                        html.H5("Upload Data", 
                               className="text-center mb-3",
                               style={'fontFamily': 'Poppins, sans-serif', 'fontWeight': '600', 'color': '#2d3748'}),
                        html.P("Upload your battery cycle data in CSV format",
                              className="text-center",
                              style={'fontSize': '14px', 'color': '#718096', 'fontFamily': 'Inter, sans-serif'})
                    ])
                ], style={**CUSTOM_STYLE['card-style'], 'border': '2px solid #e2e8f0'})
            ], md=3, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H1("2", className="text-center mb-3",
                               style={'color': '#667eea', 'fontSize': '52px', 'fontWeight': '800', 'fontFamily': 'Poppins, sans-serif'}),
                        html.H5("AI Analysis", 
                               className="text-center mb-3",
                               style={'fontFamily': 'Poppins, sans-serif', 'fontWeight': '600', 'color': '#2d3748'}),
                        html.P("Our model analyzes the data patterns",
                              className="text-center",
                              style={'fontSize': '14px', 'color': '#718096', 'fontFamily': 'Inter, sans-serif'})
                    ])
                ], style={**CUSTOM_STYLE['card-style'], 'border': '2px solid #e2e8f0'})
            ], md=3, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H1("3", className="text-center mb-3",
                               style={'color': '#667eea', 'fontSize': '52px', 'fontWeight': '800', 'fontFamily': 'Poppins, sans-serif'}),
                        html.H5("Get Predictions", 
                               className="text-center mb-3",
                               style={'fontFamily': 'Poppins, sans-serif', 'fontWeight': '600', 'color': '#2d3748'}),
                        html.P("Receive RUL predictions for all protocols",
                              className="text-center",
                              style={'fontSize': '14px', 'color': '#718096', 'fontFamily': 'Inter, sans-serif'})
                    ])
                ], style={**CUSTOM_STYLE['card-style'], 'border': '2px solid #e2e8f0'})
            ], md=3, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H1("4", className="text-center mb-3",
                               style={'color': '#667eea', 'fontSize': '52px', 'fontWeight': '800', 'fontFamily': 'Poppins, sans-serif'}),
                        html.H5("Visualize", 
                               className="text-center mb-3",
                               style={'fontFamily': 'Poppins, sans-serif', 'fontWeight': '600', 'color': '#2d3748'}),
                        html.P("View interactive charts and detailed results",
                              className="text-center",
                              style={'fontSize': '14px', 'color': '#718096', 'fontFamily': 'Inter, sans-serif'})
                    ])
                ], style={**CUSTOM_STYLE['card-style'], 'border': '2px solid #e2e8f0'})
            ], md=3, className="mb-4"),
        ], className="mb-5"),

        # Protocol Information Section
        dbc.Row([
            dbc.Col([
                html.H2("Understanding Battery Usage Protocols", 
                       className="text-center mb-4",
                       style=CUSTOM_STYLE['section-title']),
                html.P("Our system analyzes battery performance across 16 different usage patterns:",
                      className="text-center mb-4",
                      style={**CUSTOM_STYLE['body-text'], 'maxWidth': '800px', 'margin': '0 auto 30px auto'})
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div("🔋", style={'fontSize': '40px', 'marginBottom': '15px'}),
                        html.H5("Charging Patterns", 
                               className="mb-3",
                               style={'fontFamily': 'Poppins, sans-serif', 'fontWeight': '600', 'color': '#2d3748'}),
                        html.Ul([
                            html.Li("Regular, Fast, Very Fast, Super-Fast", style={'fontSize': '14px', 'marginBottom': '5px'}),
                            html.Li("Deep discharge vs. Partial charging", style={'fontSize': '14px', 'marginBottom': '5px'}),
                            html.Li("Full charge vs. Moderate charging", style={'fontSize': '14px'})
                        ], style={'color': '#718096', 'fontFamily': 'Inter, sans-serif', 'paddingLeft': '20px'})
                    ])
                ], style={**CUSTOM_STYLE['feature-card'], 'minHeight': '200px'})
            ], md=4, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div("🚗", style={'fontSize': '40px', 'marginBottom': '15px'}),
                        html.H5("Driving Behaviors", 
                               className="mb-3",
                               style={'fontFamily': 'Poppins, sans-serif', 'fontWeight': '600', 'color': '#2d3748'}),
                        html.Ul([
                            html.Li("Light, Moderate, Hard driving", style={'fontSize': '14px', 'marginBottom': '5px'}),
                            html.Li("Heavy loads and towing", style={'fontSize': '14px', 'marginBottom': '5px'}),
                            html.Li("Stop & Go city traffic", style={'fontSize': '14px'})
                        ], style={'color': '#718096', 'fontFamily': 'Inter, sans-serif', 'paddingLeft': '20px'})
                    ])
                ], style={**CUSTOM_STYLE['feature-card'], 'minHeight': '200px'})
            ], md=4, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div("⚙️", style={'fontSize': '40px', 'marginBottom': '15px'}),
                        html.H5("Combined Scenarios", 
                               className="mb-3",
                               style={'fontFamily': 'Poppins, sans-serif', 'fontWeight': '600', 'color': '#2d3748'}),
                        html.Ul([
                            html.Li("Fast charging + Deep discharge", style={'fontSize': '14px', 'marginBottom': '5px'}),
                            html.Li("Hard driving + Fast charging", style={'fontSize': '14px', 'marginBottom': '5px'}),
                            html.Li("Various mixed usage patterns", style={'fontSize': '14px'})
                        ], style={'color': '#718096', 'fontFamily': 'Inter, sans-serif', 'paddingLeft': '20px'})
                    ])
                ], style={**CUSTOM_STYLE['feature-card'], 'minHeight': '200px'})
            ], md=4, className="mb-4"),
        ], className="mb-5"),

        # Call to Action
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H3("Ready to Predict Battery Life?", 
                           className="text-center mb-4",
                           style={'fontFamily': 'Poppins, sans-serif', 'fontWeight': '700', 'fontSize': '32px', 'color': '#2d3748'}),
                    html.Div([
                        dcc.Link(
                            dbc.Button("Launch Zenodo Predictor →", 
                                      color="primary",
                                      size="lg",
                                      className="me-3",
                                      style=CUSTOM_STYLE['button-primary']),
                            href='/prediction',
                            style={'textDecoration': 'none'}
                        ),
                        dcc.Link(
                            dbc.Button("Launch NASA Predictor →", 
                                      color="primary",
                                      size="lg",
                                      style={**CUSTOM_STYLE['button-primary'], 'background': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)'}),
                            href='/nasa-prediction',
                            style={'textDecoration': 'none'}
                        )
                    ], className="text-center")
                ], className="text-center py-5",
                   style={'background': 'linear-gradient(135deg, #f6f9fc 0%, #edf2f7 100%)', 'borderRadius': '20px'})
            ], width=12)
        ])
    ], fluid=True, className="py-5", style={'backgroundColor': '#fafbfc'})

# Prediction Page Layout
def prediction_page():
    return dbc.Container([
        # Header with back button
        dbc.Row([
            dbc.Col([
                dcc.Link(
                    dbc.Button("← Back to Home", color="link", 
                              className="mb-3", 
                              style={
                                  'fontSize': '16px', 
                                  'fontFamily': 'Inter, sans-serif',
                                  'fontWeight': '500',
                                  'color': '#667eea'
                              }),
                    href='/',
                    style={'textDecoration': 'none'}
                )
            ], width=12)
        ]),
        
        # Title Section
        html.Div([
            html.H2("🔋 Battery RUL Predictor",
                    className="text-center mb-3",
                    style={
                        'fontFamily': 'Poppins, sans-serif',
                        'fontWeight': '800',
                        'color': '#2d3748',
                        'fontSize': '42px',
                        'letterSpacing': '-0.5px'
                    }),
            html.P(
                "Upload a CSV file. The app will use the last row and simulate RUL predictions for protocol IDs 1–16.",
                className="text-center mb-5",
                style={
                    'fontSize': '16px',
                    'color': '#718096',
                    'fontFamily': 'Inter, sans-serif',
                    'fontWeight': '400',
                    'maxWidth': '700px',
                    'margin': '0 auto 40px auto'
                }
            ),
        ]),

        # Upload Section
        dbc.Card([
            dbc.CardBody([
                dcc.Upload(
                    id="upload-data",
                    children=html.Div([
                        html.Div("📁", style={'fontSize': '56px', 'marginBottom': '15px'}),
                        html.Div([
                            "Drag and Drop or ",
                            html.A("Select CSV File", 
                                  style={
                                      'fontWeight': '600',
                                      'color': '#667eea',
                                      'textDecoration': 'underline',
                                      'cursor': 'pointer'
                                  })
                        ], style={
                            'fontFamily': 'Inter, sans-serif',
                            'fontSize': '18px',
                            'color': '#4a5568',
                            'fontWeight': '500'
                        }),
                        html.Div("Supported format: CSV", 
                                style={
                                    'fontSize': '13px',
                                    'color': '#a0aec0',
                                    'marginTop': '10px',
                                    'fontFamily': 'Inter, sans-serif'
                                })
                    ], style={'textAlign': 'center'}),
                    style={
                        "width": "100%",
                        "height": "180px",
                        "lineHeight": "30px",
                        "borderWidth": "2px",
                        "borderStyle": "dashed",
                        "borderRadius": '20px',
                        "textAlign": "center",
                        "padding": "20px",
                        "backgroundColor": "#f7fafc",
                        "cursor": "pointer",
                        "borderColor": "#cbd5e0",
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "center",
                        "transition": "all 0.3s ease"
                    },
                    multiple=False,
                ),
            ])
        ], style=CUSTOM_STYLE['card-style'], className="mb-4"),

        # Status message
        html.Div(id="status-msg", className="text-center mb-4",
                style={
                    'fontSize': '15px',
                    'fontWeight': '500',
                    'fontFamily': 'Inter, sans-serif',
                    'padding': '12px 20px',
                    'borderRadius': '12px'
                }),

        # Protocol Selection
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Protocol ID for Detailed RUL Prediction:",
                                  className="mb-3", 
                                  style={
                                      'fontSize': '17px',
                                      'fontWeight': '600',
                                      'fontFamily': 'Poppins, sans-serif',
                                      'color': '#2d3748'
                                  }),
                    ], width=8),
                    dbc.Col([
                        dbc.Button("📖 Protocol Guide", 
                                  id="open-protocol-modal",
                                  color="info",
                                  size="sm",
                                  outline=True,
                                  style={
                                      'fontFamily': 'Inter, sans-serif',
                                      'fontWeight': '500',
                                      'borderRadius': '20px',
                                      'fontSize': '13px'
                                  }),
                    ], width=4, className="text-end"),
                ]),
                dcc.Dropdown(
                    id="protocol-dropdown",
                    options=[
                        {"label": f"Protocol {i} - {PROTOCOL_DESCRIPTIONS[i]}", "value": i} 
                        for i in range(1, 17)
                    ],
                    value=1,
                    style={
                        "borderRadius": "12px",
                        "fontFamily": "Inter, sans-serif",
                        "fontSize": "15px"
                    },
                ),
                html.Div(id="protocol-info", className="mt-3", style={
                    'padding': '15px',
                    'backgroundColor': '#f0f4ff',
                    'borderRadius': '12px',
                    'borderLeft': '4px solid #667eea',
                    'fontFamily': 'Inter, sans-serif',
                    'fontSize': '14px',
                    'color': '#4a5568',
                    'lineHeight': '1.6'
                })
            ])
        ], style=CUSTOM_STYLE['card-style'], className="mb-4"),
        
        # Protocol Guide Modal
        dbc.Modal([
            dbc.ModalHeader(
                dbc.ModalTitle("📖 Battery Usage Protocol Guide",
                              style={'fontFamily': 'Poppins, sans-serif', 'fontWeight': '700', 'fontSize': '24px'}),
                close_button=True
            ),
            dbc.ModalBody([
                html.P("Understanding different battery usage patterns and their impact on battery life:",
                      style={'fontFamily': 'Inter, sans-serif', 'fontSize': '15px', 'color': '#718096', 'marginBottom': '25px'}),
                
                # Protocol cards in modal
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H6(f"Protocol {i}", 
                                       style={'color': '#667eea', 'fontWeight': '700', 'fontFamily': 'Poppins, sans-serif', 'marginBottom': '8px'}),
                                html.P(PROTOCOL_DETAILS[i]['title'].split(': ')[1], 
                                      style={'fontWeight': '600', 'fontSize': '15px', 'marginBottom': '8px', 'color': '#2d3748', 'fontFamily': 'Inter, sans-serif'}),
                                html.P(PROTOCOL_DETAILS[i]['description'], 
                                      style={'fontSize': '14px', 'color': '#718096', 'lineHeight': '1.6', 'marginBottom': '0', 'fontFamily': 'Inter, sans-serif'})
                            ], style={
                                'padding': '20px',
                                'backgroundColor': '#f7fafc',
                                'borderRadius': '12px',
                                'marginBottom': '15px',
                                'border': '1px solid #e2e8f0'
                            })
                        ], md=6) for i in range(1, 17)
                    ])
                ], style={'maxHeight': '500px', 'overflowY': 'auto'})
            ]),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-protocol-modal", className="ms-auto",
                          style={
                              'fontFamily': 'Inter, sans-serif',
                              'fontWeight': '500',
                              'borderRadius': '20px',
                              'padding': '8px 24px'
                          })
            ),
        ], id="protocol-modal", size="xl", is_open=False, scrollable=True),

        # Prediction Result Card
        dbc.Card([
            dbc.CardBody([
                html.H3("🎯 RUL Prediction for Selected Protocol:",
                        className="card-title text-center mb-4",
                        style={
                            'fontFamily': 'Poppins, sans-serif',
                            'fontWeight': '700',
                            'color': '#2d3748',
                            'fontSize': '26px'
                        }),
                html.Div(id="prediction-output",
                         children="--",
                         className="text-center",
                         style={
                             "fontSize": "56px", 
                             "fontWeight": "800",
                             "color": "#667eea",
                             "fontFamily": "Poppins, sans-serif",
                             "letterSpacing": "-1px"
                         }),
            ])
        ], style={**CUSTOM_STYLE['card-style'], 'background': 'linear-gradient(135deg, #f6f9fc 0%, #edf2f7 100%)'}, className="mb-4"),

        # Table of predictions
        dbc.Card([
            dbc.CardBody([
                html.H4("📋 Predicted RUL for All Protocols",
                        className="text-center mb-2",
                        style={
                            'fontFamily': 'Poppins, sans-serif',
                            'fontWeight': '700',
                            'color': '#2d3748',
                            'fontSize': '24px'
                        }),
                html.P("Compare predicted battery life across different usage patterns",
                      className="text-center mb-4",
                      style={'fontSize': '14px', 'color': '#718096', 'fontFamily': 'Inter, sans-serif'}),
                DataTable(
                    id='rul-table',
                    columns=[
                        {"name": "Protocol ID", "id": "protocol_id"},
                        {"name": "Predicted RUL (Cycles)", "id": "Predicted RUL (Cycles)"},
                    ],
                    data=[],
                    style_table={'height': '350px', 'overflowY': 'auto', 'borderRadius': '12px'},
                    style_cell={
                        'textAlign': 'center',
                        'padding': '14px',
                        'fontSize': '15px',
                        'fontFamily': 'Inter, sans-serif',
                        'fontWeight': '400'
                    },
                    style_header={
                        'backgroundColor': '#667eea',
                        'color': 'white',
                        'fontWeight': '600',
                        'fontSize': '16px',
                        'fontFamily': 'Poppins, sans-serif',
                        'border': 'none'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': '#f7fafc'
                        },
                        {
                            'if': {'row_index': 'even'},
                            'backgroundColor': 'white'
                        }
                    ],
                    tooltip_data=[],
                    tooltip_duration=None
                ),
                html.Div([
                    html.Small("💡 Tip: Higher RUL values indicate longer predicted battery life. Hover over protocols in the chart for details.",
                              style={
                                  'fontSize': '13px',
                                  'color': '#a0aec0',
                                  'fontFamily': 'Inter, sans-serif',
                                  'fontStyle': 'italic'
                              })
                ], className="text-center mt-3")
            ])
        ], style=CUSTOM_STYLE['card-style'], className="mb-4"),

        # Bar chart visualization
        dbc.Card([
            dbc.CardBody([
                dcc.Graph(id="protocol-rul-bar", style={"height": "500px"}),
            ])
        ], style=CUSTOM_STYLE['card-style']),
    ], fluid=True, className="py-4", style={'backgroundColor': '#fafbfc', 'minHeight': '100vh'})

# NASA Prediction Page Layout
def nasa_prediction_page():
    # NASA model required features
    nasa_features = [
        "voltage_v_mean",
        "current_a_mean",
        "aux_temperature_1_c_mean",
        "current_a_abs_mean",
        "rolling_mean_current_a_mean",
        "rolling_mean_aux_temperature_1_c_mean",
        "rolling_mean_current_a_abs_mean",
        "rolling_std_current_a_mean",
        "rolling_std_aux_temperature_1_c_mean",
        "rolling_std_current_a_abs_mean"
    ]
    
    return dbc.Container([
        # Header with back button
        dbc.Row([
            dbc.Col([
                dcc.Link(
                    dbc.Button("← Back to Home", color="link", 
                              className="mb-3", 
                              style={
                                  'fontSize': '16px', 
                                  'fontFamily': 'Inter, sans-serif',
                                  'fontWeight': '500',
                                  'color': '#f5576c'
                              }),
                    href='/',
                    style={'textDecoration': 'none'}
                )
            ], width=12)
        ]),
        
        # Title Section
        html.Div([
            html.H2("🚀 NASA Battery RUL Predictor",
                    className="text-center mb-3",
                    style={
                        'fontFamily': 'Poppins, sans-serif',
                        'fontWeight': '800',
                        'color': '#2d3748',
                        'fontSize': '42px',
                        'letterSpacing': '-0.5px'
                    }),
            html.P(
                "Fine-tuned NASA model for accurate RUL prediction. Enter values manually or upload a CSV file.",
                className="text-center mb-5",
                style={
                    'fontSize': '16px',
                    'color': '#718096',
                    'fontFamily': 'Inter, sans-serif',
                    'fontWeight': '400',
                    'maxWidth': '700px',
                    'margin': '0 auto 40px auto'
                }
            ),
        ]),

        # Input Mode Selection
        dbc.Card([
            dbc.CardBody([
                html.Label("Select Input Mode:",
                          className="mb-3", 
                          style={
                              'fontSize': '17px',
                              'fontWeight': '600',
                              'fontFamily': 'Poppins, sans-serif',
                              'color': '#2d3748'
                          }),
                dbc.RadioItems(
                    id="nasa-input-mode",
                    options=[
                        {"label": " 📝 Manual Input", "value": "manual"},
                        {"label": " 📁 CSV Upload", "value": "csv"},
                    ],
                    value="manual",
                    inline=True,
                    style={'fontSize': '15px', 'fontFamily': 'Inter, sans-serif'}
                ),
            ])
        ], style=CUSTOM_STYLE['card-style'], className="mb-4"),

        # Manual Input Section
        html.Div(id="nasa-manual-input-section", children=[
            dbc.Card([
                dbc.CardBody([
                    html.H5("Enter Battery Features Manually",
                           className="mb-4",
                           style={'fontFamily': 'Poppins, sans-serif', 'fontWeight': '600', 'color': '#2d3748'}),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Voltage Mean (V)", style={'fontSize': '14px', 'fontWeight': '500'}),
                            dbc.Input(id="input-voltage_v_mean", type="number", placeholder="e.g., 3.7", step=0.01),
                        ], md=6, className="mb-3"),
                        dbc.Col([
                            dbc.Label("Current Mean (A)", style={'fontSize': '14px', 'fontWeight': '500'}),
                            dbc.Input(id="input-current_a_mean", type="number", placeholder="e.g., 1.5", step=0.01),
                        ], md=6, className="mb-3"),
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Temperature Mean (°C)", style={'fontSize': '14px', 'fontWeight': '500'}),
                            dbc.Input(id="input-aux_temperature_1_c_mean", type="number", placeholder="e.g., 25.0", step=0.1),
                        ], md=6, className="mb-3"),
                        dbc.Col([
                            dbc.Label("Current Absolute Mean (A)", style={'fontSize': '14px', 'fontWeight': '500'}),
                            dbc.Input(id="input-current_a_abs_mean", type="number", placeholder="e.g., 1.5", step=0.01),
                        ], md=6, className="mb-3"),
                    ]),
                    
                    html.Hr(className="my-4"),
                    html.H6("Rolling Statistics Features", 
                           className="mb-3",
                           style={'fontFamily': 'Poppins, sans-serif', 'fontWeight': '600', 'color': '#667eea'}),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Rolling Mean Current (A)", style={'fontSize': '14px', 'fontWeight': '500'}),
                            dbc.Input(id="input-rolling_mean_current_a_mean", type="number", placeholder="e.g., 1.5", step=0.01),
                        ], md=6, className="mb-3"),
                        dbc.Col([
                            dbc.Label("Rolling Mean Temperature (°C)", style={'fontSize': '14px', 'fontWeight': '500'}),
                            dbc.Input(id="input-rolling_mean_aux_temperature_1_c_mean", type="number", placeholder="e.g., 25.0", step=0.1),
                        ], md=6, className="mb-3"),
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Rolling Mean Current Abs (A)", style={'fontSize': '14px', 'fontWeight': '500'}),
                            dbc.Input(id="input-rolling_mean_current_a_abs_mean", type="number", placeholder="e.g., 1.5", step=0.01),
                        ], md=6, className="mb-3"),
                        dbc.Col([
                            dbc.Label("Rolling Std Current (A)", style={'fontSize': '14px', 'fontWeight': '500'}),
                            dbc.Input(id="input-rolling_std_current_a_mean", type="number", placeholder="e.g., 0.2", step=0.01),
                        ], md=6, className="mb-3"),
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Rolling Std Temperature (°C)", style={'fontSize': '14px', 'fontWeight': '500'}),
                            dbc.Input(id="input-rolling_std_aux_temperature_1_c_mean", type="number", placeholder="e.g., 1.0", step=0.1),
                        ], md=6, className="mb-3"),
                        dbc.Col([
                            dbc.Label("Rolling Std Current Abs (A)", style={'fontSize': '14px', 'fontWeight': '500'}),
                            dbc.Input(id="input-rolling_std_current_a_abs_mean", type="number", placeholder="e.g., 0.2", step=0.01),
                        ], md=6, className="mb-3"),
                    ]),
                    
                    html.Div([
                        dbc.Button("🔮 Predict RUL", 
                                  id="nasa-predict-button",
                                  color="primary",
                                  size="lg",
                                  className="mt-4",
                                  style={**CUSTOM_STYLE['button-primary'], 'background': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)'})
                    ], className="text-center")
                ])
            ], style=CUSTOM_STYLE['card-style'], className="mb-4")
        ]),

        # CSV Upload Section
        html.Div(id="nasa-csv-input-section", children=[
            dbc.Card([
                dbc.CardBody([
                    dcc.Upload(
                        id="nasa-upload-data",
                        children=html.Div([
                            html.Div("📁", style={'fontSize': '56px', 'marginBottom': '15px'}),
                            html.Div([
                                "Drag and Drop or ",
                                html.A("Select CSV File", 
                                      style={
                                          'fontWeight': '600',
                                          'color': '#f5576c',
                                          'textDecoration': 'underline',
                                          'cursor': 'pointer'
                                      })
                            ], style={
                                'fontFamily': 'Inter, sans-serif',
                                'fontSize': '18px',
                                'color': '#4a5568',
                                'fontWeight': '500'
                            }),
                            html.Div("File should contain the required NASA features", 
                                    style={
                                        'fontSize': '13px',
                                        'color': '#a0aec0',
                                        'marginTop': '10px',
                                        'fontFamily': 'Inter, sans-serif'
                                    })
                        ], style={'textAlign': 'center'}),
                        style={
                            "width": "100%",
                            "height": "180px",
                            "lineHeight": "30px",
                            "borderWidth": "2px",
                            "borderStyle": "dashed",
                            "borderRadius": '20px',
                            "textAlign": "center",
                            "padding": "20px",
                            "backgroundColor": "#fff5f7",
                            "cursor": "pointer",
                            "borderColor": "#f5576c",
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "transition": "all 0.3s ease"
                        },
                        multiple=False,
                    ),
                ])
            ], style=CUSTOM_STYLE['card-style'], className="mb-4"),
        ], style={'display': 'none'}),

        # Status message
        html.Div(id="nasa-status-msg", className="text-center mb-4",
                style={
                    'fontSize': '15px',
                    'fontWeight': '500',
                    'fontFamily': 'Inter, sans-serif',
                    'padding': '12px 20px',
                    'borderRadius': '12px'
                }),

        # Prediction Result Card
        dbc.Card([
            dbc.CardBody([
                html.H3("🎯 NASA RUL Prediction Result:",
                        className="card-title text-center mb-4",
                        style={
                            'fontFamily': 'Poppins, sans-serif',
                            'fontWeight': '700',
                            'color': '#2d3748',
                            'fontSize': '26px'
                        }),
                html.Div(id="nasa-prediction-output",
                         children="--",
                         className="text-center",
                         style={
                             "fontSize": "56px", 
                             "fontWeight": "800",
                             "color": "#f5576c",
                             "fontFamily": "Poppins, sans-serif",
                             "letterSpacing": "-1px"
                         }),
            ])
        ], style={**CUSTOM_STYLE['card-style'], 'background': 'linear-gradient(135deg, #fff5f7 0%, #ffeef2 100%)'}, className="mb-4"),

        # Feature Info Card
        dbc.Card([
            dbc.CardBody([
                html.H5("📊 Required NASA Features",
                       className="mb-3",
                       style={'fontFamily': 'Poppins, sans-serif', 'fontWeight': '600', 'color': '#2d3748'}),
                html.P("The NASA model requires the following 10 features:",
                      style={'fontSize': '14px', 'color': '#718096', 'fontFamily': 'Inter, sans-serif'}),
                html.Ul([
                    html.Li(feat, style={'fontSize': '13px', 'marginBottom': '5px', 'fontFamily': 'monospace'})
                    for feat in nasa_features
                ], style={'color': '#4a5568'})
            ])
        ], style=CUSTOM_STYLE['card-style']),
    ], fluid=True, className="py-4", style={'backgroundColor': '#fafbfc', 'minHeight': '100vh'})

# Main layout with page container
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='navigation-store'),  # Store for navigation state
    html.Div(id='page-content')
])

# Inject custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background-color: #fafbfc;
            }
            
            .hover-lift:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 24px rgba(0,0,0,0.12) !important;
            }
            
            .Select-control {
                border-radius: 12px !important;
                border-color: #cbd5e0 !important;
                font-family: 'Inter', sans-serif !important;
            }
            
            .Select-menu-outer {
                border-radius: 12px !important;
                font-family: 'Inter', sans-serif !important;
            }
            
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5) !important;
            }
            
            .dash-table-container {
                border-radius: 12px !important;
                overflow: hidden;
            }
            
            /* Upload area hover effect */
            div[data-dash-is-loading] {
                opacity: 1 !important;
            }
            
            /* Smooth transitions */
            * {
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }
            
            /* Custom scrollbar */
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }
            
            ::-webkit-scrollbar-track {
                background: #f1f1f1;
                border-radius: 10px;
            }
            
            ::-webkit-scrollbar-thumb {
                background: #cbd5e0;
                border-radius: 10px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: #667eea;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Page routing callback
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/prediction':
        return prediction_page()
    elif pathname == '/nasa-prediction':
        return nasa_prediction_page()
    else:
        return landing_page()

# Modal toggle callback
@app.callback(
    Output("protocol-modal", "is_open"),
    [Input("open-protocol-modal", "n_clicks"), Input("close-protocol-modal", "n_clicks")],
    [State("protocol-modal", "is_open")],
    prevent_initial_call=True
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

# Protocol info display callback
@app.callback(
    Output("protocol-info", "children"),
    Input("protocol-dropdown", "value"),
    prevent_initial_call=False
)
def display_protocol_info(protocol_id):
    if protocol_id is None:
        return ""
    
    protocol_detail = PROTOCOL_DETAILS.get(protocol_id, {})
    title = protocol_detail.get('title', '')
    description = protocol_detail.get('description', '')
    
    return html.Div([
        html.Div([
            html.Strong("ℹ️ Selected Protocol: ", style={'color': '#667eea', 'fontSize': '15px'}),
            html.Span(title, style={'fontWeight': '600', 'color': '#2d3748'})
        ], style={'marginBottom': '8px'}),
        html.Div(description, style={'fontSize': '14px', 'color': '#4a5568'})
    ])

# Prediction callback
@app.callback(
    Output("prediction-output", "children"),
    Output("status-msg", "children"),
    Output("rul-table", "data"),
    Output("protocol-rul-bar", "figure"),
    Input("upload-data", "contents"),
    Input("protocol-dropdown", "value"),
    State("upload-data", "filename"),
)
def update_output(contents, selected_protocol, filename):
    if contents is None:
        return (
            "--",
            html.Div("📤 Upload a CSV file containing battery cycle data to begin.", 
                    style={'color': '#6c757d'}),
            [],
            {},
        )

    try:
        # Decode uploaded CSV
        content_type, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)
        try:
            s = decoded.decode("utf-8")
        except UnicodeDecodeError:
            s = decoded.decode("ISO-8859-1")

        df_raw = pd.read_csv(io.StringIO(s))
        if df_raw.empty:
            return (
                "--", 
                html.Div("❌ Uploaded file is empty or unreadable.", 
                        style={'color': '#dc3545'}), 
                [], 
                {}
            )

        # Use only the last row
        last_row = df_raw.iloc[[-1]]

        protocol_rul_list = []

        # Simulate RUL for all protocols
        for protocol in range(1, 17):
            row_copy = last_row.copy()
            row_copy["protocol_id"] = protocol
            preds = predictor_zenodo.predict(row_copy)
            rul_value = float(preds[0])
            protocol_rul_list.append({
                "protocol_id": protocol,
                "Protocol Label": f"Protocol {protocol}",
                "Predicted RUL (Cycles)": rul_value,
                "Description": PROTOCOL_DESCRIPTIONS[protocol]
            })

        protocol_rul_df = pd.DataFrame(protocol_rul_list)

        # Find selected protocol RUL
        selected_rul = protocol_rul_df.loc[
            protocol_rul_df["protocol_id"] == selected_protocol,
            "Predicted RUL (Cycles)"
        ].values[0]
        prediction_text = f"{selected_rul:.2f} cycles"

        status = html.Div([
            html.I(className="bi bi-check-circle-fill me-2", style={'color': '#28a745'}),
            html.Span([
                f"✅ Successfully analyzed file: ",
                html.Strong(filename),
                f". Simulated using the last row for protocols 1–16."
            ])
        ], style={'color': '#28a745'})

        # Bar chart with distinct colors and labels
        bar_fig = px.bar(
            protocol_rul_df,
            x="Protocol Label",
            y="Predicted RUL (Cycles)",
            color="Protocol Label",  # unique color for each bar
            text="Predicted RUL (Cycles)",
            title="<b>Predicted RUL by Protocol ID</b>",
            color_discrete_sequence=px.colors.qualitative.Set3,
            hover_data={"Description": True, "Protocol Label": False, "Predicted RUL (Cycles)": ":.2f"}
        )

        bar_fig.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>%{customdata[0]}<br>RUL: <b>%{y:.2f} cycles</b><extra></extra>",
            marker=dict(
                line=dict(width=1.5, color='white'),
                opacity=0.9
            ),
            textfont=dict(size=13, family="Inter, sans-serif", color="#2d3748", weight=600)
        )
        bar_fig.update_layout(
            xaxis_title="<b>Protocol</b>",
            yaxis_title="<b>Predicted RUL (Cycles)</b>",
            uniformtext_minsize=8,
            uniformtext_mode='hide',
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
            hovermode="x unified",
            font=dict(size=14, family="Inter, sans-serif", color="#4a5568"),
            title_font=dict(size=22, color='#2d3748', family="Poppins, sans-serif", weight=700),
            title_x=0.5,
            margin=dict(t=70, b=60, l=60, r=40),
            xaxis=dict(
                showgrid=False,
                linecolor='#e2e8f0',
                linewidth=2,
                tickfont=dict(size=13, family="Inter, sans-serif", color="#4a5568")
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='#f7fafc',
                linecolor='#e2e8f0',
                linewidth=2,
                tickfont=dict(size=13, family="Inter, sans-serif", color="#4a5568")
            )
        )

        return prediction_text, status, protocol_rul_df.to_dict("records"), bar_fig

    except Exception as e:
        print(f"Error: {e}")
        return (
            "--",
            html.Div([
                f"❌ Error processing file: ",
                html.Strong(filename),
                html.Br(),
                html.Small(f"Details: {str(e)}", style={'fontSize': '12px'})
            ], style={'color': '#dc3545'}),
            [],
            {},
        )


# NASA Input Mode Toggle Callback
@app.callback(
    Output("nasa-manual-input-section", "style"),
    Output("nasa-csv-input-section", "style"),
    Input("nasa-input-mode", "value"),
)
def toggle_nasa_input_mode(mode):
    if mode == "manual":
        return {'display': 'block'}, {'display': 'none'}
    else:
        return {'display': 'none'}, {'display': 'block'}


# NASA Manual Prediction Callback
@app.callback(
    Output("nasa-prediction-output", "children"),
    Output("nasa-status-msg", "children"),
    Input("nasa-predict-button", "n_clicks"),
    Input("nasa-upload-data", "contents"),
    State("input-voltage_v_mean", "value"),
    State("input-current_a_mean", "value"),
    State("input-aux_temperature_1_c_mean", "value"),
    State("input-current_a_abs_mean", "value"),
    State("input-rolling_mean_current_a_mean", "value"),
    State("input-rolling_mean_aux_temperature_1_c_mean", "value"),
    State("input-rolling_mean_current_a_abs_mean", "value"),
    State("input-rolling_std_current_a_mean", "value"),
    State("input-rolling_std_aux_temperature_1_c_mean", "value"),
    State("input-rolling_std_current_a_abs_mean", "value"),
    State("nasa-upload-data", "filename"),
    prevent_initial_call=True
)
def nasa_predict(n_clicks, csv_contents, voltage, current, temp, current_abs,
                 roll_mean_current, roll_mean_temp, roll_mean_current_abs,
                 roll_std_current, roll_std_temp, roll_std_current_abs, filename):
    
    ctx = callback_context
    if not ctx.triggered:
        return "--", ""
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    try:
        # Handle CSV upload
        if trigger_id == "nasa-upload-data" and csv_contents is not None:
            content_type, content_string = csv_contents.split(",", 1)
            decoded = base64.b64decode(content_string)
            try:
                s = decoded.decode("utf-8")
            except UnicodeDecodeError:
                s = decoded.decode("ISO-8859-1")
            
            df_raw = pd.read_csv(io.StringIO(s))
            if df_raw.empty:
                return "--", html.Div("❌ Uploaded file is empty.", style={'color': '#dc3545'})
            
            # Use the last row
            last_row = df_raw.iloc[[-1]]
            
            # Predict
            predictions = predictor_nasa.predict(last_row)
            rul_value = float(predictions[0])
            
            prediction_text = f"{rul_value:.2f} cycles"
            status = html.Div([
                f"✅ Successfully predicted from file: ",
                html.Strong(filename)
            ], style={'color': '#28a745'})
            
            return prediction_text, status
        
        # Handle manual input
        elif trigger_id == "nasa-predict-button":
            # Validate all inputs are provided
            inputs = [voltage, current, temp, current_abs, roll_mean_current, 
                     roll_mean_temp, roll_mean_current_abs, roll_std_current, 
                     roll_std_temp, roll_std_current_abs]
            
            if any(x is None for x in inputs):
                return "--", html.Div("❌ Please fill in all feature values.", style={'color': '#dc3545'})
            
            # Create DataFrame with manual inputs
            input_data = pd.DataFrame({
                "voltage_v_mean": [voltage],
                "current_a_mean": [current],
                "aux_temperature_1_c_mean": [temp],
                "current_a_abs_mean": [current_abs],
                "rolling_mean_current_a_mean": [roll_mean_current],
                "rolling_mean_aux_temperature_1_c_mean": [roll_mean_temp],
                "rolling_mean_current_a_abs_mean": [roll_mean_current_abs],
                "rolling_std_current_a_mean": [roll_std_current],
                "rolling_std_aux_temperature_1_c_mean": [roll_std_temp],
                "rolling_std_current_a_abs_mean": [roll_std_current_abs]
            })
            
            # Predict
            predictions = predictor_nasa.predict(input_data)
            rul_value = float(predictions[0])
            
            prediction_text = f"{rul_value:.2f} cycles"
            status = html.Div("✅ Prediction completed successfully!", style={'color': '#28a745'})
            
            return prediction_text, status
        
        return "--", ""
    
    except Exception as e:
        print(f"NASA Prediction Error: {e}")
        return "--", html.Div([
            "❌ Error during prediction: ",
            html.Br(),
            html.Small(str(e), style={'fontSize': '12px'})
        ], style={'color': '#dc3545'})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)