#include <gmock/gmock.h>

#include "xgboostpredictor.h"


namespace xgboost::predictor
{

//------------------------------------------------------------------------------

TEST(XGBoostPredictor, Predict)
{
    XGBoostPredictor predictor("data/info.model.json");

    XGBoostPredictor::Data data(240);
    for (size_t i = 0; i < data.size(); i += 2)
    {
        data[i] = 2 * i - 14.58f;
    }

    const auto prediction = predictor.predict(data, true);
    ASSERT_EQ(prediction.size(), 1U);
    ASSERT_FLOAT_EQ(prediction[0], -1.6755048f);
}

//------------------------------------------------------------------------------

TEST(XGBoostPredictor, Rank)
{
    XGBoostPredictor predictor("data/info.model.json"); // "binary:logistic"

    std::vector<XGBoostPredictor::Data> candidates;
    for (size_t i = 0; i < 2; ++i)
    {
        candidates.emplace_back(240);
        auto& data = candidates.back();
        for (size_t j = i; j < data.size(); j += 2)
        {
            data[j] = 2 * i - 14.58f + 3 * j;
        }
    }

    auto scores = predictor.predict(candidates, false);
    ASSERT_EQ(scores.size(), 2U);
    ASSERT_FLOAT_EQ(scores[0], 0.15769163f);
    ASSERT_FLOAT_EQ(scores[1], 0.18086274f);
}

//------------------------------------------------------------------------------ 

TEST(XGBoostPredictor, Transform)
{
    // empty input
    {
        std::vector<float> v;
        XGBoostPredictor::transform(v, XGBoostPredictor::Transformation::SOFTMAX);
        ASSERT_TRUE(v.empty());
    }

    // sigmoid
    {
        std::vector<float> v({0.0f});
        XGBoostPredictor::transform(v, XGBoostPredictor::Transformation::SIGMOID);
        ASSERT_EQ(v.size(), 1U);
        ASSERT_FLOAT_EQ(v[0], 0.5f);
    }
    {
        std::vector<float> v({1.0f});
        XGBoostPredictor::transform(v, XGBoostPredictor::Transformation::SIGMOID);
        ASSERT_EQ(v.size(), 1U);
        ASSERT_FLOAT_EQ(v[0], 0.7310586f);
    }
    // softmax
    {
        std::vector<float> v({0.0f, 0.0f});
        XGBoostPredictor::transform(v, XGBoostPredictor::Transformation::SOFTMAX);
        ASSERT_EQ(v.size(), 2U);
        ASSERT_FLOAT_EQ(v[0], 0.5f);
        ASSERT_FLOAT_EQ(v[1], 0.5f);
    }
    {
        std::vector<float> v({11.0f, 11.0f});
        XGBoostPredictor::transform(v, XGBoostPredictor::Transformation::SOFTMAX);
        ASSERT_EQ(v.size(), 2U);
        ASSERT_FLOAT_EQ(v[0], 0.5f);
        ASSERT_FLOAT_EQ(v[1], 0.5f);
    }
    {
        std::vector<float> v({-11.43f, 14.28f, 0.23f});
        XGBoostPredictor::transform(v, XGBoostPredictor::Transformation::SOFTMAX);
        ASSERT_EQ(v.size(), 3U);
        ASSERT_FLOAT_EQ(v[0], 6.827928e-12f);
        ASSERT_FLOAT_EQ(v[1], 0.99999923f);
        ASSERT_FLOAT_EQ(v[2], 7.9097379e-07f);
    }
}

//------------------------------------------------------------------------------

TEST(XGBoostPredictor, FileDoesNotExist)
{
    ASSERT_THROW(XGBoostPredictor("foo.bar"), std::runtime_error);
}

//------------------------------------------------------------------------------

} // namespaces
