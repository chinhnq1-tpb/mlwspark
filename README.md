# bscore_rebuild

## Build FLow

![Flow Diagram](/images/build_flow.png)

## Project Structure

```css
mlwspark/
├─ .dvc/
├─ configs/
├─ images/
├─ notebooks/
├─ src/
│  ├── CustomModels/
│  ├── Evaluators/
│  ├── FeatureSelectors/
│  ├── FeatureTransformers/
│  ├── utils/
├─ .dvcignore
├─ .gitignore
├─ README.md
└─ environment.yml
```

- `configs`: Contains project's configuration files
- `notebooks`: Contains Jupyter notebooks
- `src/CustomModels`: Contains SparkML custom Models: e.g Passive Agressive
- `src/Evaluators`: Contains SparkML custom Evaluator: e.g Gini Index
- `src/FeatureSelectors`: Contains SparkML custom Feature Selectors: e.g MRMR Algorithm
- `src/FeatureTransformers`: Contains SparkML custom Feature Transformer: e.g WOE Transformer
- `src/utils`: Contains ultility functions

## Setup

> [!Prerequisite]
> - `conda` == 25.7.0

