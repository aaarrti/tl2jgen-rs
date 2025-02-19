use anyhow::Ok;
use serde::Deserialize;

#[derive(Debug, Clone, Copy)]
pub struct Leaf {
    pub node_id: u32,
    pub leaf_value: f64,
}

#[derive(Debug, Clone)]
pub struct TestNode {
    pub node_id: u32,
    pub split_feature_id: u32,
    pub comparison_op: String,
    pub threshold: f64,
    pub left_child: u32,
    pub right_child: u32,
}

#[derive(Debug, Clone)]
pub enum Node {
    Leaf(Leaf),
    TestNode(TestNode),
}

#[derive(Clone, Debug, Deserialize)]
struct UnknownNode {
    node_id: u32,
    leaf_value: Option<f64>,
    split_feature_id: Option<u32>,
    default_left: Option<bool>,
    comparison_op: Option<String>,
    threshold: Option<f64>,
    left_child: Option<u32>,
    right_child: Option<u32>,
    node_type: Option<String>,
    has_categorical_split: Option<bool>,
}

#[derive(Clone, Debug, Deserialize)]

pub struct Tree<T> {
    pub nodes: Vec<T>,
}

#[derive(Debug, Clone, Deserialize)]
struct ModelJSON {
    leaf_output_type: String,
    num_feature: u32,
    task_type: String,
    average_tree_output: bool,
    num_target: u32,
    trees: Vec<Tree<UnknownNode>>,
}

#[derive(Debug, Clone)]
pub struct ModelIR {
    pub num_features: u32,
    pub trees: Vec<Tree<Node>>,
}

impl TryFrom<&UnknownNode> for Node {
    type Error = anyhow::Error;

    fn try_from(value: &UnknownNode) -> Result<Self, Self::Error> {
        if let Some(true) = value.has_categorical_split {
            anyhow::bail!("Unsupported categorical split");
        }

        if let Some(node_type) = value.node_type.as_deref() {
            if node_type != "numerical_test_node" {
                anyhow::bail!("Not supported node type: {}", node_type);
            }

            if let Some(false) = value.default_left {
                anyhow::bail!("Not supported default left");
            }

            Ok(Node::TestNode(TestNode {
                node_id: value.node_id,
                split_feature_id: value.split_feature_id.unwrap(),
                comparison_op: value.comparison_op.as_deref().unwrap().to_string(),
                threshold: value.threshold.unwrap(),
                left_child: value.left_child.unwrap(),
                right_child: value.right_child.unwrap(),
            }))
        } else {
            Ok(Node::Leaf(Leaf {
                node_id: value.node_id,
                leaf_value: value.leaf_value.unwrap(),
            }))
        }
    }
}

impl TryFrom<&Tree<UnknownNode>> for Tree<Node> {
    type Error = anyhow::Error;

    fn try_from(value: &Tree<UnknownNode>) -> Result<Self, Self::Error> {
        let nodes: anyhow::Result<Vec<Node>> = value.nodes.iter().map(|i| i.try_into()).collect();
        let nodes = nodes?;
        Ok(Tree { nodes })
    }
}

pub mod json {

    use super::{ModelIR, ModelJSON, Node, Tree};
    use anyhow::Context;
    use anyhow::Result;

    pub fn decode(path: &str) -> Result<ModelIR> {
        tracing::info!("Parsing JSON from {}...", path);
        let file = std::fs::File::open(path).with_context(|| "Failed to open file")?;
        let reader = std::io::BufReader::with_capacity(1024 * 1024, file);
        let json: ModelJSON =
            serde_json::from_reader(reader).with_context(|| "Faiuled to read JSON")?;
        anyhow::ensure!(json.average_tree_output, "Unsupported average tree output");
        anyhow::ensure!(
            json.leaf_output_type == "float64",
            std::format!("Unsupported leaf output type: {}", json.leaf_output_type)
        );
        anyhow::ensure!(
            json.num_target == 1,
            std::format!("Unsupported num target: {}", json.num_target)
        );
        anyhow::ensure!(
            json.task_type == "kRegressor",
            std::format!("Unsupported task type: {}", json.task_type)
        );
        tracing::debug!("Converting nodes ...");
        let trees: Result<Vec<Tree<Node>>> = json.trees.iter().map(|i| i.try_into()).collect();
        let trees = trees?;
        Ok(ModelIR {
            num_features: json.num_feature,
            trees,
        })
    }
}

mod binary {

    use super::ModelIR;
    use anyhow::Result;

    pub fn decode(path: &str) -> Result<ModelIR> {
        anyhow::bail!("Not implemented")
    }
}
