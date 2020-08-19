#pragma once

#include <rapidjson/fwd.h>

#include <string>
#include <vector>
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/istreamwrapper.h>
#include <unordered_set>

#include <fstream>
#include <cmath>

#ifdef USE_EXPERIMENTAL_OPTIONAL
#include <experimental/optional>
#else
#include <optional>
#endif

namespace xgboost::predictor
{

//------------------------------------------------------------------------------

// thread-safe XGBoost predictor
class XGBoostPredictor
{
public:
    // output margin transformations
    enum class Transformation
    {
        NONE,
        SIGMOID,
        SOFTMAX
    };

    // sparse data array, vector item index is feature index
#ifdef USE_EXPERIMENTAL_OPTIONAL
    using Data = std::vector<std::experimental::optional<float>>;
#else
    using Data = std::vector<std::optional<float>>;
#endif

    //------------------------------------------------------------------------------
    // create predictor from JSON model file
    //------------------------------------------------------------------------------
    XGBoostPredictor(const std::string& jsonFile)
        :m_model(parse(jsonFile))
    {}

    //------------------------------------------------------------------------------
    // make prediction
    //------------------------------------------------------------------------------
    std::vector<float> predict(const Data& data, const bool outputMargin = false) const
    {
        std::vector<float> predictions;
        predictions.reserve(m_model.predictors.size());

        for (const auto& predictor : m_model.predictors)
        {
            predictions.push_back(predict(data, predictor));
        }

        if (!outputMargin)
        {
            transform(predictions, m_model.transformation);
        }

        return predictions;
    }

    //------------------------------------------------------------------------------
    // make multiple predictions
    //------------------------------------------------------------------------------
    std::vector<float> predict(const std::vector<Data>& data, const bool outputMargin = false) const
    {
        if (m_model.predictors.size() != 1)
        {
            throw std::runtime_error("xgboost predict incompatible model size: " + std::to_string(m_model.predictors.size()));
        }

        std::vector<float> scores;
        scores.reserve(data.size());

        for (const auto& d: data)
        {
            scores.push_back(predict(d, m_model.predictors.front()));
        }

        if (!outputMargin)
        {
            transform(scores, m_model.transformation);
        }

        return scores;
    }

    //------------------------------------------------------------------------------
    // output margin transformation according to the objective
    //------------------------------------------------------------------------------
    static void transform(std::vector<float>& predictions, const Transformation transformation)
    {
        if (predictions.empty())
        {
            return;
        }

        if (transformation == Transformation::SIGMOID)
        {
            transformSigmoid(predictions);
        }
        else if (transformation == Transformation::SOFTMAX)
        {
            transformSoftmax(predictions);
        }
    }

    //------------------------------------------------------------------------------
    // sigmoid transformation
    //------------------------------------------------------------------------------
    static void transformSigmoid(std::vector<float>& predictions)
    {
        for (float& prediction: predictions)
        {
            prediction = 1.0f / (1.0f + expf(-prediction));
        }
    }

    //------------------------------------------------------------------------------
    // softmax transformation
    //------------------------------------------------------------------------------
    static void transformSoftmax(std::vector<float>& predictions)
    {
        // softmax transformation
        float max = std::numeric_limits<float>::min();
        for (const float prediction : predictions)
        {
            max = std::max(max, prediction);
        }
        double sum = 0.0;
        for (float& prediction : predictions)
        {
            prediction = expf(prediction - max);
            sum += prediction;
        }
        for (float& prediction : predictions)
        {
            prediction /= sum;
        }
    }


private:
    // decision node of tree
    struct Node
    {
        float value = 0.0f;         // threshold value for decision node, leaf value for leaf node
        int feature = -1;           // feature index for decision node, -1 for leaf node
        unsigned int yes = 0;       // if (feature < value) next node index = yes
        unsigned int no = 0;        // if (feature >= value) next node index = no
        unsigned int missing = 0;   // if (feature is missing) next node index = missing
    };

    // tree
    using Tree = std::vector<Node>;

    // predictor
    using Predictor = std::vector<Tree>;

    // model: multiple predictors for multiclass prediction
    //        transformed base score (according to the objective)
    struct Model
    {
        const std::vector<Predictor> predictors;
        const float base_score = 0.0f;
        const Transformation transformation = Transformation::NONE;
    };


private:
    //------------------------------------------------------------------------------
    // calculate prediction
    //------------------------------------------------------------------------------
    float predict(const Data& data, const Predictor& predictor) const
    {
        float prediction = 0.0f;

        for (const auto& tree : predictor)
        {
            prediction += predict(data, tree);
        }

        prediction += m_model.base_score;

        return prediction;
    }

    //------------------------------------------------------------------------------
    // calculate tree prediction
    //------------------------------------------------------------------------------
    float predict(const Data& data, const Tree& tree) const
    {
        const auto size = data.size();

        unsigned int index = 0;

        while (true)
        {
            const Node& node = tree[index];

            if (node.feature < 0)
            {
                return node.value;
            }

            if (static_cast<unsigned int>(node.feature) < size && data[node.feature])
            {
                index = *data[node.feature] < node.value ? node.yes : node.no;
            }
            else
            {
                index = node.missing;
            }
        }
    }

    //------------------------------------------------------------------------------
    // parse JSON model file
    //------------------------------------------------------------------------------
    static Model parse(const std::string& jsonFile)
    {
        // stream wrapper
        std::ifstream stream(jsonFile);
        rapidjson::IStreamWrapper wrapper(stream);

        // parse JSON model
        rapidjson::Document doc;
        if (doc.ParseStream(wrapper).HasParseError() || !doc.IsObject())
        {
            throw std::runtime_error("invalid xgboost json model");
        }

        // get json object
        auto getObject = [](const auto& value, const char* key)
        {
            if (!value.HasMember(key) || !value[key].IsObject())
            {
                throw std::runtime_error("missing or invalid json value member: " + std::string(key));
            }
            return value[key].GetObject();
        };

        // get json array
        auto getArray = [](const auto& value, const char* key)
        {
            if (!value.HasMember(key) || !value[key].IsArray())
            {
                throw std::runtime_error("missing or invalid json array member: " + std::string(key));
            }
            return value[key].GetArray();
        };

        // get bool array
        auto getArrayBool = [&getArray](const auto& value, const char* key)
        {
            std::vector<bool> result;

            for (const auto& member : getArray(value, key))
            {
                if (member.IsBool())
                {
                    result.emplace_back(member.GetBool());
                }
                else
                {
                    throw std::runtime_error(std::string(key) + " json array member is not bool");
                }
            }

            return result;
        };

        // get int array
        auto getArrayInt = [&getArray](const auto& value, const char* key)
        {
            std::vector<int> result;

            for (const auto& member : getArray(value, key))
            {
                if (member.IsInt())
                {
                    result.emplace_back(member.GetInt());
                }
                else
                {
                    throw std::runtime_error(std::string(key) + " json array member is not int");
                }
            }

            return result;
        };

        // get float array
        auto getArrayFloat = [&getArray](const auto& value, const char* key)
        {
            std::vector<float> result;

            for (const auto& member : getArray(value, key))
            {
                if (member.IsDouble())
                {
                    result.emplace_back(member.GetDouble());
                }
                else if (member.IsInt64())
                {
                    result.emplace_back(member.GetInt64());
                }
                else
                {
                    throw std::runtime_error(std::string(key) + " json array member is not double/int");
                }
            }

            return result;
        };

        // get json string
        auto getString = [](const auto& value, const char* key)
        {
            if (!value.HasMember(key) || !value[key].IsString())
            {
                throw std::runtime_error("missing or invalid json string member: " + std::string(key));
            }
            return std::string(value[key].GetString());
        };

        // get learner
        const auto& learner = getObject(doc, "learner");

        // get gradient booster
        const auto& gradient_booster = getObject(learner, "gradient_booster");

        // get model
        const auto& model = getObject(gradient_booster, "model");

        // parse trees
        Predictor predictor;

        for (const auto& json_tree : getArray(model, "trees"))
        {
            const auto default_left = getArrayBool(json_tree, "default_left");
            const auto left_children = getArrayInt(json_tree, "left_children");
            const auto right_children = getArrayInt(json_tree, "right_children");
            const auto split_indices = getArrayInt(json_tree, "split_indices");
            const auto split_conditions = getArrayFloat(json_tree, "split_conditions");

            checkSizes(default_left.size(), left_children.size(), right_children.size(), split_indices.size(), split_conditions.size());

            predictor.emplace_back();
            auto& tree = predictor.back();

            for (size_t i = 0; i < default_left.size(); ++i)
            {
                tree.emplace_back();
                auto& node = tree.back();
                node.value = split_conditions[i];
                node.feature = left_children[i] >= 0 ? split_indices[i] : -1;
                node.yes = left_children[i];
                node.no = right_children[i];
                node.missing = default_left[i] ? left_children[i] : right_children[i];
            }

            check(tree);
        }

        // get tree_info for multiclass predictors
        const auto tree_info = getArrayInt(model, "tree_info");
        if (tree_info.size() != predictor.size())
        {
            throw std::runtime_error("unexprected tree_info size: " + std::to_string(tree_info.size()) +
                    ", trees: " + std::to_string(predictor.size()));
        }

        // build predictors
        std::vector<Predictor> predictors;

        for (size_t i = 0; i < tree_info.size(); ++i)
        {
            const int group = tree_info[i];
            if (group < 0)
            {
                throw std::runtime_error("unexpected tree_info group: " + std::to_string(group));
            }

            predictors.resize(predictors.size() < group + 1U ? group + 1U : predictors.size());

            predictors[group].emplace_back(predictor[i]);
        }

        // get objective and raw base score
        const auto objective = getString(getObject(learner, "objective"), "name");
        const auto base_score = std::stof(getString(getObject(learner, "learner_model_param"), "base_score"));

        // transform base score according to the objective
        auto transformBaseScore = [objective, base_score]()
        {
            if (objective == "reg:logistic" || objective == "binary:logistic" || objective == "binary:logitraw")
            {
                if (base_score <= 0.0f || base_score >= 1.0f)
                {
                    throw std::runtime_error("base_score must be in (0,1) for logistic loss, got: " + std::to_string(base_score));
                }
                return -std::log(1.0f / base_score - 1.0f);
            }

            if (objective == "reg:gamma" || objective == "reg:tweedie" || objective == "count:poisson" || objective == "survival:aft" || objective == "survival:cox")
            {
                return std::log(base_score);
            }

            return base_score;
        };

        // get transformation according to the objective
        auto transformation = [objective]()
        {
            if (objective ==  "multi:softprob")
            {
                return Transformation::SOFTMAX;
            }
            else if (objective == "reg:logistic" || objective == "binary:logistic")
            {
                return Transformation::SIGMOID;
            }
            return Transformation::NONE;
        };

        return Model{std::move(predictors), transformBaseScore(), transformation()};
    }

    //------------------------------------------------------------------------------
    // check tree is valid
    //------------------------------------------------------------------------------
    static void check(const Tree& tree)
    {
        if (tree.empty())
        {
            throw std::runtime_error("empty tree");
        }

        // check all indices are in range
        for (const auto& node : tree)
        {
            if (node.feature >= 0)
            {
                if (node.yes > tree.size() || node.no > tree.size() || node.missing > tree.size())
                {
                    throw std::runtime_error("tree yes/no/missing index out of range");
                }
            }
        }

        // check there are no cycles
        std::vector<bool> visited(tree.size());
        check(tree, 0, visited);
    }

    //------------------------------------------------------------------------------
    // check tree has no cycles
    //------------------------------------------------------------------------------
    static void check(const Tree& tree, const size_t index, std::vector<bool>& visited)
    {
        const auto& node = tree[index];

        if (node.feature >= 0)
        {
            if (visited[index])
            {
                throw std::runtime_error("cycle in tree");
            }
            visited[index] = true;

            check(tree, node.yes, visited);
            if (node.no != node.yes)
            {
                check(tree, node.no, visited);
            }
            if (node.missing != node.yes && node.missing != node.no)
            {
                check(tree, node.missing, visited);
            }
        }
    }

    //------------------------------------------------------------------------------
    // check sizes are the same
    //------------------------------------------------------------------------------
    template<typename ...Sizes>
    static void checkSizes(const Sizes&... sizes)
    {
#ifdef USE_EXPERIMENTAL_OPTIONAL
        std::experimental::optional<size_t> size;
#else
        std::optional<size_t> size;
#endif
        const size_t args[] { sizes... };
        for (size_t i = 0; i < sizeof...(sizes); ++i)
        {
            if (!size)
            {
                size = args[i];
            }
            else
            {
                if (*size != args[i])
                {
                    throw std::runtime_error("json array sizes do not match");
                }
            }
        }
    }


private:
    const Model m_model;
};

//------------------------------------------------------------------------------

} // namespaces
