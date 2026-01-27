
import json
import os

notebook_path = "/home/master/Documents/Python _Classes/fraud-with-SMOTE.ipynb"

if not os.path.exists(notebook_path):
    print(f"Error: Notebook not found at {notebook_path}")
    exit(1)

# Load existing notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Filter out old deployment cells to avoid duplicates
print("Cleaning old deployment cells...")
original_count = len(nb['cells'])
cleaned_cells = []
for cell in nb['cells']:
    source = "".join(cell.get('source', []))
    # Remove cells that look like the ones we added previously
    if "## 8. Deployment" in source or "SAVING DEPLOYMENT ARTIFACTS" in source or "streamlit run app.py" in source:
        continue
    cleaned_cells.append(cell)

nb['cells'] = cleaned_cells
print(f"Removed {original_count - len(cleaned_cells)} old cells.")

# Define New Cells (With Forced LightGBM Selection)
new_cells = [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Deployment Preparation\n",
    "\n",
    "This section saves the necessary artifacts (model, features, stats) for the external Streamlit application."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import joblib\n",
    "import json\n",
    "\n",
    "print(\"=\" * 70)\n",
    "print(\"SAVING DEPLOYMENT ARTIFACTS\")\n",
    "print(\"=\" * 70)\n",
    "\n",
    "# 1. Select Best LightGBM Model (Narrative Aligned)\n",
    "# User confirmed they settled for LightGBM (optimized via PR-curve/Class Weight) \n",
    "# and that SMOTE had little impact. To match the paper text, we prefer the Class Weight version.\n",
    "lgbm_cw = master_df[(master_df['Model'] == 'LightGBM') & (master_df['Technique'] == 'Class Weight')]\n",
    "\n",
    "if not lgbm_cw.empty:\n",
    "    # There should only be one, but we take the best if multiple\n",
    "    best_row = lgbm_cw.sort_values('PR-AUC', ascending=False).iloc[0]\n",
    "    best_model_name = best_row['Model']\n",
    "    best_technique = best_row['Technique']\n",
    "    \n",
    "    print(f\"Saving Model (Narrative Aligned): {best_model_name} ({best_technique})\")\n",
    "    print(f\"Performance: PR-AUC={best_row['PR-AUC']:.4f}, F1={best_row['F1']:.4f}\")\n",
    "\n",
    "    # For Class Weight models, the object is in 'models' dict\n",
    "    final_model = models[best_model_name]\n",
    "\n",
    "    # Save Model\n",
    "    joblib.dump(final_model, 'model.pkl')\n",
    "    print(\"✅ Model saved to model.pkl\")\n",
    "    \n",
    "elif not master_df[master_df['Model'] == 'LightGBM'].empty:\n",
    "    # Fallback if Class Weight version missing for some reason\n",
    "    print(\"⚠️ Class Weight version not found. Using best available LightGBM.\")\n",
    "    best_row = master_df[master_df['Model'] == 'LightGBM'].sort_values('PR-AUC', ascending=False).iloc[0]\n",
    "    best_model_name = best_row['Model']\n",
    "    best_technique = best_row['Technique']\n",
    "    \n",
    "    print(f\"Saving Model (Fallback): {best_model_name} ({best_technique})\")\n",
    "    if best_technique == 'SMOTE':\n",
    "        final_model = smote_models[best_model_name]\n",
    "    else:\n",
    "        final_model = models[best_model_name]\n",
    "        \n",
    "    joblib.dump(final_model, 'model.pkl')\n",
    "    print(\"✅ Model saved to model.pkl\")\n",
    "else:\n",
    "    print(\"❌ LightGBM model not found in results!\")\n",
    "\n",
    "# 2. Save Feature Columns\n",
    "with open('feature_cols.json', 'w') as f:\n",
    "    json.dump(feature_cols, f)\n",
    "print(f\"✅ Feature list saved to feature_cols.json ({len(feature_cols)} features)\")\n",
    "\n",
    "# 3. Save Account Stats\n",
    "account_stats.to_csv('account_stats_artifact.csv')\n",
    "print(\"✅ Account stats saved to account_stats_artifact.csv\")\n",
    "\n",
    "# 4. Save Fill Values (Medians)\n",
    "fill_values = {}\n",
    "cols_to_fill = account_stats.columns\n",
    "for col in cols_to_fill:\n",
    "    if 'count' in col:\n",
    "        fill_values[col] = 0\n",
    "    else:\n",
    "        fill_values[col] = train_fe[col].median()\n",
    "\n",
    "# Convert numpy types for JSON\n",
    "for k, v in fill_values.items():\n",
    "    if hasattr(v, 'item'):\n",
    "        fill_values[k] = v.item()\n",
    "\n",
    "with open('fill_values.json', 'w') as f:\n",
    "    json.dump(fill_values, f)\n",
    "print(\"✅ Fill values (medians) saved to fill_values.json\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Launching the App\n",
    "\n",
    "Now that artifacts are saved, run the Streamlit app from your terminal:\n",
    "\n",
    "```bash\n",
    "streamlit run app.py\n",
    "```"
   ]
  }
]

nb['cells'].extend(new_cells)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
    
print("Successfully replaced artifact saving cells in notebook.")
