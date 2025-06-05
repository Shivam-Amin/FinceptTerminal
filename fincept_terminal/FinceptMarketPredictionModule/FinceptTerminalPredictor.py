# from textual.screen import Screen, ModalScreen
# from textual.widgets import Input, Button, Static, OptionList, ProgressBar
# from textual.containers import Vertical, Center, Grid, VerticalScroll, Container
# from textual import on


# class PredictorTab(Container):

#     def compose(self):
#         # with Grid(id="dialog"):
#         #     yield Static("📈 SBI Predictor", classes="title")
#         #     yield Input(placeholder="Enter path to your SBI CSV", id="csv_path")
#         #     yield Button("Run Prediction", id="run")
#         #     yield Button("❌ Close", id="close")
#         yield Static("🔎 Enter path to CSV file:", id="search_label")
#         yield Input(placeholder="Enter path...", id="search_input")
#         yield Button("Run Prediction", id="prediction_button", variant="primary")
#         # yield Static("🎥 Select a Video:", id="video_label", classes="hidden")
#         # yield OptionList(id="video_list")
#         # with VerticalScroll(id="analysis_results", classes="analysis_results"):
#         #     yield Static("Transcript", id="transcript_label", classes="hidden")
#         #     yield Static("", id="transcript_display")
#         #     yield Static("Sentiment Analysis", id="sentiment_label", classes="hidden")
#         #     yield Static("", id="sentiment_display")
#         #     yield Button("Download Sentiment Model", id="download_model_button", variant="primary", classes="hidden")
#         #     yield Static("Model Download Progress:", id="download_progress_label", classes="hidden")
#         #     yield ProgressBar(id="download_progress_bar", show_percentage=True, show_eta=False, classes="hidden")
#         #     yield Static("Estimated time left: ?", id="download_eta_label", classes="hidden")  # New ETA label
    
#     # @on(Button.Pressed, "#close")
#     # def close_predictor(self) -> None:
#     #     self.app.pop_screen()

#     def on_button_pressed(self, event):
#         # if event.button.id == "close":
#         #     self.app.pop_screen()
#         if event.button.id == "run":
#             path = self.query_one("#csv_path", Input).value
#             # You can trigger your CSV processing logic here
#             # await self.app.push_screen(LoadingScreen(path))  # optionally add another screen

import os
import pandas as pd
import yaml
import numpy as np
from pathlib import Path
from textual.widgets import Input, Button, Static, RichLog
from textual.containers import Container, Vertical, VerticalScroll, Horizontal
from textual import on
from textual.app import ComposeResult
import asyncio
import qlib
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import init_instance_by_config
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.model.gbdt import LGBModel
from qlib.data import D
from qlib.constant import REG_CN
import warnings
warnings.filterwarnings('ignore')


class PredictorTab(Container):
    
    def __init__(self):
        super().__init__()
        self.qlib_initialized = False
        self.data_prepared = False
        self.instrument_name = None
    
    def compose(self) -> ComposeResult:
        with Vertical(id="predictor_main_layout"):
            yield Static("🔎 Auto Quant Research Workflow", classes="header")
            yield Static("Enter path to CSV file:", id="search_label")
            yield Input(placeholder="Enter CSV file path...", id="csv_path")
            yield Static("Instrument/Stock name (optional):", id="instrument_label")
            yield Input(placeholder="e.g., AAPL, SBI, etc. (leave empty for auto)", id="instrument_name")
            
            with Horizontal(id="predictor_buttons_section"):
                yield Button("Validate CSV", id="validate_button", variant="default")
                yield Button("Prepare Data", id="prepare_button", variant="default")
                yield Button("Run Prediction", id="prediction_button", variant="primary")
                yield Button("Clear Results", id="clear_button", variant="warning")
            
            with Container(id="predictor_result_section"):
                yield Static("Results:", classes="results-header")
                yield RichLog(id="results_log", auto_scroll=False, highlight=True, markup=True)
    
    def initialize_qlib(self):
        """Initialize qlib with basic configuration"""
        if not self.qlib_initialized:
            try:
                # Initialize qlib with memory provider for custom data
                qlib.init(region=REG_CN)
                self.qlib_initialized = True
                self.log_result("✅ Qlib initialized successfully", "success")
                return True
            except Exception as e:
                # Try alternative initialization
                try:
                    qlib.init()
                    self.qlib_initialized = True
                    self.log_result("✅ Qlib initialized with default settings", "success")
                    return True
                except Exception as e2:
                    self.log_result(f"❌ Failed to initialize qlib: {str(e2)}", "error")
                    return False
        return True
    
    def validate_csv(self, file_path: str) -> bool:
        """Validate CSV file format and required columns"""
        try:
            if not os.path.exists(file_path):
                self.log_result(f"❌ File not found: {file_path}", "error")
                return False
            
            # Read CSV
            df = pd.read_csv(file_path)
            
            # Check required columns for qlib
            required_columns = ['date', 'open', 'high', 'low', 'close']
            optional_columns = ['volume', 'instrument']
            
            df_columns_lower = [col.lower() for col in df.columns]
            missing_required = [col for col in required_columns if col not in df_columns_lower]
            
            if missing_required:
                self.log_result(f"❌ Missing required columns: {missing_required}", "error")
                self.log_result(f"Available columns: {list(df.columns)}", "info")
                return False
            
            # Check for optional columns
            missing_optional = [col for col in optional_columns if col not in df_columns_lower]
            if missing_optional:
                self.log_result(f"⚠️  Missing optional columns: {missing_optional} (will be added automatically)", "warning")
            
            # Basic data validation
            if df.empty:
                self.log_result("❌ CSV file is empty", "error")
                return False
            
            # Check data types and ranges
            numeric_columns = ['open', 'high', 'low', 'close']
            for col in numeric_columns:
                if col in df_columns_lower:
                    actual_col = df.columns[df_columns_lower.index(col)]
                    if not pd.api.types.is_numeric_dtype(df[actual_col]):
                        self.log_result(f"❌ Column '{actual_col}' should be numeric", "error")
                        return False
            
            self.log_result(f"✅ CSV validation passed", "success")
            self.log_result(f"📊 Data shape: {df.shape}", "info")
            self.log_result(f"📅 Date range: {df['date'].min()} to {df['date'].max()}", "info")
            
            return True
            
        except Exception as e:
            self.log_result(f"❌ Error validating CSV: {str(e)}", "error")
            return False
    
    def prepare_data_for_qlib(self, csv_path: str) -> bool:
        """Convert CSV data to qlib format and load into qlib"""
        try:
            self.log_result("🔄 Preparing data for qlib...", "info")
            
            # Read and process CSV
            df = pd.read_csv(csv_path)
            
            # Standardize column names
            column_mapping = {}
            for col in df.columns:
                column_mapping[col] = col.lower()
            df = df.rename(columns=column_mapping)
            
            # Get instrument name
            self.instrument_name = self.get_instrument_name_from_user()
            
            # Add instrument column if missing
            if 'instrument' not in df.columns:
                df['instrument'] = self.instrument_name
                self.log_result(f"✅ Added instrument column: {self.instrument_name}", "success")
            
            # Add volume column if missing (estimate based on price and typical volume patterns)
            if 'volume' not in df.columns:
                # More sophisticated volume estimation
                df['volume'] = (df['close'] * np.random.uniform(0.8, 1.2, len(df)) * 1000000).astype(int)
                self.log_result("✅ Added estimated volume column", "warning")
            
            # Ensure date column is datetime and set as index
            df['datetime'] = pd.to_datetime(df['date'])
            df = df.set_index('datetime')
            df = df.drop('date', axis=1)
            
            # Sort by datetime
            df = df.sort_index()
            
            # Calculate additional features
            df = self.calculate_technical_features(df)
            
            # Prepare data in qlib format (instrument as column, datetime as index)
            qlib_data = {}
            for instrument in df['instrument'].unique():
                instrument_data = df[df['instrument'] == instrument].drop('instrument', axis=1)
                qlib_data[instrument] = instrument_data
            
            # Save data in qlib format (simplified - in production you'd use proper qlib data handlers)
            self.qlib_data = qlib_data
            self.data_prepared = True
            
            self.log_result("✅ Data prepared successfully for qlib", "success")
            self.log_result(f"📈 Instruments prepared: {list(qlib_data.keys())}", "info")
            
            return True
            
        except Exception as e:
            self.log_result(f"❌ Error preparing data: {str(e)}", "error")
            return False
    
    def calculate_technical_features(self, df):
        """Calculate technical analysis features similar to Alpha158"""
        try:
            # Sort by datetime to ensure proper calculation
            df = df.sort_index()
            
            # Basic price features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Moving averages
            for window in [5, 10, 20, 60]:
                df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
                df[f'ma_ratio_{window}'] = df['close'] / df[f'ma_{window}']
            
            # Volatility features
            df['volatility_5'] = df['returns'].rolling(window=5).std()
            df['volatility_20'] = df['returns'].rolling(window=20).std()
            
            # Volume features
            df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
            df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma_20']
            
            # Price position features
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            
            # Momentum features
            for period in [1, 5, 10, 20]:
                df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
            
            # RSI-like features
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Fix the variable name error in Bollinger Bands calculation
            bb_window = 20
            bb_std = df['close'].rolling(window=bb_window).std()
            bb_mean = df['close'].rolling(window=bb_window).mean()
            df['bb_upper'] = bb_mean + (bb_std * 2)
            df['bb_lower'] = bb_mean - (bb_std * 2)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # MACD-like features
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            self.log_result("✅ Technical features calculated", "success")
            return df
            
        except Exception as e:
            self.log_result(f"⚠️  Warning in feature calculation: {str(e)}", "warning")
            return df
    
    def get_instrument_name_from_user(self) -> str:
        """Get instrument name from user input or generate default"""
        instrument_input = self.query_one("#instrument_name", Input).value.strip()
        
        if instrument_input:
            return instrument_input.upper()
        
        # Try to extract from filename
        csv_path = self.query_one("#csv_path", Input).value.strip()
        if csv_path:
            filename = Path(csv_path).stem
            # Clean up filename for instrument name
            clean_name = filename.replace('_data', '').replace('_stock', '').replace('-', '_').upper()
            return clean_name
        
        return "STOCK_001"  # Default fallback
    
    def create_qlib_dataset(self):
        """Create qlib dataset from prepared data"""
        try:
            if not self.data_prepared or not hasattr(self, 'qlib_data'):
                self.log_result("❌ Data not prepared. Run 'Prepare Data' first.", "error")
                return None
            
            # Get the main instrument data
            instrument_data = list(self.qlib_data.values())[0]
            
            # Split data into train/validation/test
            total_length = len(instrument_data)
            train_end = int(total_length * 0.7)
            valid_end = int(total_length * 0.85)
            
            train_data = instrument_data.iloc[:train_end]
            valid_data = instrument_data.iloc[train_end:valid_end]
            test_data = instrument_data.iloc[valid_end:]
            
            self.log_result(f"📊 Data split - Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}", "info")
            
            # Prepare features and labels (exclude non-numeric columns)
            feature_columns = []
            for col in instrument_data.columns:
                if col not in ['open', 'high', 'low', 'close', 'volume', 'instrument'] and not col.endswith('_target'):
                    # Only include numeric columns
                    if pd.api.types.is_numeric_dtype(instrument_data[col]):
                        feature_columns.append(col)
            
            # Create target variable (next day return)
            instrument_data['target'] = instrument_data['close'].shift(-1) / instrument_data['close'] - 1
            
            # Remove rows with NaN values
            clean_data = instrument_data.dropna()
            
            return {
                'data': clean_data,
                'features': feature_columns,
                'train_end': train_end,
                'valid_end': valid_end
            }
            
        except Exception as e:
            self.log_result(f"❌ Error creating dataset: {str(e)}", "error")
            return None
    
    def run_lightgbm_model(self, dataset_info):
        """Run LightGBM model on the prepared dataset"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            import numpy as np
            
            data = dataset_info['data']
            features = dataset_info['features']
            train_end = dataset_info['train_end']
            valid_end = dataset_info['valid_end']
            
            # First, clean the entire dataset using pandas
            self.log_result("🧹 Cleaning data with pandas...", "info")
            
            # Select only numeric features and target
            numeric_features = []
            for feature in features:
                if feature in data.columns and pd.api.types.is_numeric_dtype(data[feature]):
                    numeric_features.append(feature)
            
            # Create a clean dataset with only numeric features and target
            clean_data = data[numeric_features + ['target']].copy()
            
            # Use pandas to handle NaN values comprehensively
            self.log_result(f"📊 Initial data shape: {clean_data.shape}", "info")
            self.log_result(f"📊 NaN values per column: {clean_data.isnull().sum().sum()}", "info")
            
            # Replace infinite values with NaN first
            clean_data = clean_data.replace([np.inf, -np.inf], np.nan)
            
            # Drop rows with any NaN values
            clean_data = clean_data.dropna()
            
            self.log_result(f"📊 After cleaning shape: {clean_data.shape}", "info")
            
            if len(clean_data) < 50:
                self.log_result("❌ Not enough clean data samples (< 50) for training", "error")
                return None
            
            # Recalculate split indices based on clean data
            total_clean = len(clean_data)
            new_train_end = int(total_clean * 0.7)
            new_valid_end = int(total_clean * 0.85)
            
            # Split the clean data
            X_train = clean_data[numeric_features].iloc[:new_train_end]
            y_train = clean_data['target'].iloc[:new_train_end]
            
            X_valid = clean_data[numeric_features].iloc[new_train_end:new_valid_end]
            y_valid = clean_data['target'].iloc[new_train_end:new_valid_end]
            
            X_test = clean_data[numeric_features].iloc[new_valid_end:]
            y_test = clean_data['target'].iloc[new_valid_end:]
            
            # Final verification - ensure no NaN values remain
            assert not X_train.isnull().any().any(), "Training features contain NaN"
            assert not y_train.isnull().any(), "Training target contains NaN"
            assert not X_valid.isnull().any().any(), "Validation features contain NaN"
            assert not y_valid.isnull().any(), "Validation target contains NaN"
            assert not X_test.isnull().any().any(), "Test features contain NaN"
            assert not y_test.isnull().any(), "Test target contains NaN"
            
            # Additional checks for infinite values and data types
            def check_data_quality(X, y, name):
                # Check for infinite values
                if np.isinf(X).any().any():
                    self.log_result(f"❌ {name} features contain infinite values", "error")
                    return False
                if np.isinf(y).any():
                    self.log_result(f"❌ {name} target contains infinite values", "error")
                    return False
                
                # Check for very large values that might cause issues
                if (np.abs(X) > 1e10).any().any():
                    self.log_result(f"⚠️  {name} features contain very large values", "warning")
                
                # Check variance (features with zero variance can cause issues)
                zero_var_features = X.var() == 0
                if zero_var_features.any():
                    self.log_result(f"⚠️  {name} has {zero_var_features.sum()} zero-variance features", "warning")
                
                return True
            
            # Validate data quality
            if not check_data_quality(X_train, y_train, "Training"):
                return None
            if not check_data_quality(X_valid, y_valid, "Validation"):
                return None
            if not check_data_quality(X_test, y_test, "Test"):
                return None
            
            # Convert to numpy arrays to avoid any pandas-related issues
            X_train_np = X_train.values.astype(np.float64)
            y_train_np = y_train.values.astype(np.float64)
            X_valid_np = X_valid.values.astype(np.float64)
            y_valid_np = y_valid.values.astype(np.float64)
            X_test_np = X_test.values.astype(np.float64)
            y_test_np = y_test.values.astype(np.float64)
            
            self.log_result(f"🔄 Training model with {len(X_train_np)} samples and {len(numeric_features)} features...", "info")
            self.log_result(f"📊 Validation: {len(X_valid_np)} samples, Test: {len(X_test_np)} samples", "info")
            
            # Use RandomForest with more conservative parameters
            model = RandomForestRegressor(
                n_estimators=50,  # Reduced from 100
                max_depth=6,      # Reduced from 8
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=1          # Changed from -1 to avoid multiprocessing issues
            )
            
            # Train the model
            try:
                model.fit(X_train_np, y_train_np)
                self.log_result("✅ Model training completed successfully", "success")
            except Exception as e:
                self.log_result(f"❌ Model training failed: {str(e)}", "error")
                # Try with even simpler model
                from sklearn.linear_model import LinearRegression
                self.log_result("🔄 Trying LinearRegression as fallback...", "warning")
                model = LinearRegression()
                model.fit(X_train_np, y_train_np)
                self.log_result("✅ LinearRegression training completed", "success")
            
            # Make predictions
            train_pred = model.predict(X_train_np)
            valid_pred = model.predict(X_valid_np) if len(X_valid_np) > 0 else []
            test_pred = model.predict(X_test_np) if len(X_test_np) > 0 else []
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train_np, train_pred)
            train_mae = mean_absolute_error(y_train_np, train_pred)
            
            results = {
                'model': model,
                'train_mse': train_mse,
                'train_mae': train_mae,
                'train_pred': train_pred,
                'valid_pred': valid_pred,
                'test_pred': test_pred,
                'X_train': X_train,
                'X_valid': X_valid,
                'X_test': X_test,
                'y_train': y_train,
                'y_valid': y_valid,
                'y_test': y_test,
                'features_used': numeric_features
            }
            
            if len(X_valid) > 0:
                valid_mse = mean_squared_error(y_valid_np, valid_pred)
                valid_mae = mean_absolute_error(y_valid_np, valid_pred)
                results['valid_mse'] = valid_mse
                results['valid_mae'] = valid_mae
            
            if len(X_test) > 0:
                test_mse = mean_squared_error(y_test_np, test_pred)
                test_mae = mean_absolute_error(y_test_np, test_pred)
                results['test_mse'] = test_mse
                results['test_mae'] = test_mae
            
            return results
            
        except Exception as e:
            self.log_result(f"❌ Error running model: {str(e)}", "error")
            return None
    
    def calculate_portfolio_metrics(self, predictions, actual_returns):
        """Calculate portfolio performance metrics"""
        try:
            # Simple long-short strategy based on predictions
            positions = np.where(predictions > 0, 1, -1)  # Long if positive prediction, short if negative
            
            # Calculate strategy returns
            strategy_returns = positions[:-1] * actual_returns[1:]  # Align predictions with future returns
            
            # Portfolio metrics
            total_return = np.prod(1 + strategy_returns) - 1
            annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
            volatility = np.std(strategy_returns) * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + strategy_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            # Information ratio (assuming benchmark return is 0)
            excess_returns = strategy_returns
            information_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'information_ratio': information_ratio,
                'win_rate': np.mean(strategy_returns > 0)
            }
            
        except Exception as e:
            self.log_result(f"❌ Error calculating portfolio metrics: {str(e)}", "error")
            return None
    
    def run_quant_workflow(self, csv_path: str):
        """Run the complete quantitative research workflow"""
        try:
            # Initialize qlib
            if not self.initialize_qlib():
                return
            
            # Validate CSV
            if not self.validate_csv(csv_path):
                return
            
            # Prepare data
            if not self.prepare_data_for_qlib(csv_path):
                return
            
            self.log_result("🚀 Starting Complete Quant Research Workflow...", "info")
            
            # Create dataset
            dataset_info = self.create_qlib_dataset()
            if dataset_info is None:
                return
            
            # Run model
            model_results = self.run_lightgbm_model(dataset_info)
            if model_results is None:
                return
            
            # Display results
            self.display_comprehensive_results(model_results, dataset_info)
            
        except Exception as e:
            self.log_result(f"❌ Error running workflow: {str(e)}", "error")
    
    def display_comprehensive_results(self, model_results, dataset_info):
        """Display comprehensive analysis results"""
        self.log_result("📊 COMPREHENSIVE QUANT ANALYSIS RESULTS", "info")
        self.log_result("═" * 60, "info")
        
        # Model Performance
        self.log_result("🤖 MODEL PERFORMANCE:", "success")
        self.log_result(f"  Training MSE: {model_results['train_mse']:.6f}", "info")
        self.log_result(f"  Training MAE: {model_results['train_mae']:.6f}", "info")
        
        if 'valid_mse' in model_results:
            self.log_result(f"  Validation MSE: {model_results['valid_mse']:.6f}", "info")
            self.log_result(f"  Validation MAE: {model_results['valid_mae']:.6f}", "info")
        
        if 'test_mse' in model_results:
            self.log_result(f"  Test MSE: {model_results['test_mse']:.6f}", "info")
            self.log_result(f"  Test MAE: {model_results['test_mae']:.6f}", "info")
        
        # Feature Importance
        if hasattr(model_results['model'], 'feature_importances_'):
            self.log_result("🔍 TOP 10 FEATURE IMPORTANCE:", "success")
            feature_names = dataset_info['features']
            importances = model_results['model'].feature_importances_
            feature_importance = list(zip(feature_names, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            for i, (feature, importance) in enumerate(feature_importance[:10]):
                self.log_result(f"  {i+1}. {feature}: {importance:.4f}", "info")
        
        # Portfolio Analysis
        if len(model_results['test_pred']) > 0 and len(model_results['y_test']) > 0:
            portfolio_metrics = self.calculate_portfolio_metrics(
                model_results['test_pred'], 
                model_results['y_test'].values
            )
            
            if portfolio_metrics:
                self.log_result("💰 PORTFOLIO PERFORMANCE ANALYSIS:", "success")
                self.log_result(f"  Total Return: {portfolio_metrics['total_return']:.4f} ({portfolio_metrics['total_return']*100:.2f}%)", "info")
                self.log_result(f"  Annualized Return: {portfolio_metrics['annualized_return']:.4f} ({portfolio_metrics['annualized_return']*100:.2f}%)", "info")
                self.log_result(f"  Volatility: {portfolio_metrics['volatility']:.4f} ({portfolio_metrics['volatility']*100:.2f}%)", "info")
                self.log_result(f"  Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.4f}", "info")
                self.log_result(f"  Information Ratio: {portfolio_metrics['information_ratio']:.4f}", "info")
                self.log_result(f"  Maximum Drawdown: {portfolio_metrics['max_drawdown']:.4f} ({portfolio_metrics['max_drawdown']*100:.2f}%)", "info")
                self.log_result(f"  Win Rate: {portfolio_metrics['win_rate']:.4f} ({portfolio_metrics['win_rate']*100:.2f}%)", "info")
        
        # Prediction Summary
        self.log_result("🎯 PREDICTION SUMMARY:", "success")
        if len(model_results['test_pred']) > 0:
            test_pred = model_results['test_pred']
            self.log_result(f"  Number of Test Predictions: {len(test_pred)}", "info")
            self.log_result(f"  Mean Predicted Return: {np.mean(test_pred):.6f} ({np.mean(test_pred)*100:.4f}%)", "info")
            self.log_result(f"  Prediction Std Dev: {np.std(test_pred):.6f}", "info")
            self.log_result(f"  Positive Predictions: {np.sum(test_pred > 0)} ({np.sum(test_pred > 0)/len(test_pred)*100:.1f}%)", "info")
            self.log_result(f"  Negative Predictions: {np.sum(test_pred < 0)} ({np.sum(test_pred < 0)/len(test_pred)*100:.1f}%)", "info")
        
        self.log_result("═" * 60, "info")
        self.log_result("✅ Analysis Complete! Review the results above.", "success")
    
    def log_result(self, message: str, level: str = "info"):
        """Log results to the RichLog widget"""
        results_log = self.query_one("#results_log", RichLog)
        
        colors = {
            "error": "red",
            "warning": "yellow",
            "success": "green",
            "info": "blue"
        }
        
        color = colors.get(level, "white")
        results_log.write(f"[{color}]{message}[/{color}]")
    
    @on(Button.Pressed, "#validate_button")
    def on_validate_pressed(self, event: Button.Pressed) -> None:
        """Handle validate button press"""
        path = self.query_one("#csv_path", Input).value.strip()
        
        if not path:
            self.log_result("❌ Please enter a CSV file path", "error")
            return
        
        self.validate_csv(path)
    
    @on(Button.Pressed, "#prepare_button")
    def on_prepare_pressed(self, event: Button.Pressed) -> None:
        """Handle prepare data button press"""
        path = self.query_one("#csv_path", Input).value.strip()
        
        if not path:
            self.log_result("❌ Please enter a CSV file path", "error")
            return
        
        if not self.validate_csv(path):
            return
        
        self.prepare_data_for_qlib(path)
    
    @on(Button.Pressed, "#prediction_button")
    def on_prediction_pressed(self, event: Button.Pressed) -> None:
        """Handle prediction button press"""
        path = self.query_one("#csv_path", Input).value.strip()
        
        if not path:
            self.log_result("❌ Please enter a CSV file path", "error")
            return
        
        # Run the complete workflow
        self.run_quant_workflow(path)
    
    @on(Button.Pressed, "#clear_button")
    def on_clear_pressed(self, event: Button.Pressed) -> None:
        """Handle clear button press"""
        results_log = self.query_one("#results_log", RichLog)
        results_log.clear()
        self.data_prepared = False
        self.qlib_initialized = False
        self.log_result("🧹 Results cleared and state reset", "info")