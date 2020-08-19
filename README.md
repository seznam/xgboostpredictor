# XGBoostPredictor
C++ header-only thread-safe library of [XGBoost](https://github.com/dmlc/xgboost/) predictor without dependency on xgboost library. 

## Requirements
``` 
C++17 compiler
rapidjson library
model serialized to json format (requires xgboost >= 1.0)
```

## Using library in C++

```cpp
#include "xgboostpredictor.h"
#include <iostream>

using namespace xgboost::predictor;

int main()
{
    // load xgboost json model
    XGBoostPredictor predictor("model.json");

    // prepare features (3 features total)
    XGBoostPredictor::Data data(3);

    // set features
    // NOTE: feature 0 is set to 1.2, feature 1 is missing and feature 2 is set to 3.4
    data[0] = 1.2f;
    data[2] = 3.4f;
    
    // make prediction
    const auto prediction = predictor.predict(data);

    // print predicted value
    std::cout << prediction[0] << std::endl;
}
```
