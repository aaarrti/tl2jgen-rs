use std::collections::HashMap;

use anyhow::Result;
use clap::Parser;
use rayon::prelude::*;
use std::io::Write;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Clone, Parser)]
pub struct Cli {
    #[arg(short, long)]
    source: String,
    #[arg(short, long)]
    destination: String,
    #[arg(short, long)]
    package: String,
}

fn setup_tracing() -> Result<()> {
    let subscriber = tracing_subscriber::fmt()
        .pretty()
        .with_env_filter(EnvFilter::from_default_env())
        .with_thread_ids(true)
        .with_max_level(tracing::Level::INFO)
        .finish();

    tracing::subscriber::set_global_default(subscriber)?;
    Ok(())
}

mod parser {

    use serde::Deserialize;

    #[derive(Clone, Debug, Deserialize)]
    pub struct Tree {
        pub nodes: Vec<UnknownNode>,
        pub has_categorical_split: bool,
    }

    #[derive(Clone, Debug, Deserialize)]
    pub struct UnknownNode {
        pub node_id: u32,
        pub leaf_value: Option<f64>,
        pub split_feature_id: Option<u32>,
        // pub default_left: Option<bool>,
        pub comparison_op: Option<String>,
        pub threshold: Option<f64>,
        pub left_child: Option<u32>,
        pub right_child: Option<u32>,
        pub node_type: Option<String>,
        pub has_categorical_split: Option<bool>,
        pub category_list: Option<Vec<u32>>,
        pub category_list_right_child: Option<bool>
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct ModelJSON {
        pub leaf_output_type: String,
        pub num_feature: u32,
        pub task_type: String,
        pub average_tree_output: bool,
        pub num_target: u32,
        pub trees: Vec<Tree>,
        pub base_scores: Vec<f64>,
    }
}

#[derive(Debug, Clone, Copy)]
struct Leaf {
    node_id: u32,
    leaf_value: f64,
}

#[derive(Debug, Clone)]
struct TestNode {
    node_id: u32,
    split_feature_id: u32,
    comparison_op: String,
    threshold: f64,
    left_child: u32,
    right_child: u32,
}

/// category_list_right_child = true
#[derive(Debug, Clone)]
struct CategoricalTestNode {
    node_id: u32,
    left_child: u32,
    right_child: u32,
    split_feature_id: u32,
    category_list: Vec<u32>,
}

#[derive(Debug, Clone)]
enum Node {
    Leaf(Leaf),
    TestNode(TestNode),
    Categorical(CategoricalTestNode),
}

impl Node {
    fn id(&self) -> u32 {
        match self {
            Node::Leaf(leaf) => leaf.node_id,
            Node::TestNode(test_node) => test_node.node_id,
            Node::Categorical(cat) => cat.node_id,
        }
    }
}

#[derive(Debug, Clone)]
struct Tree {
    inner: HashMap<u32, Node>,
    has_categorical_split: bool,
}

impl Tree {
    fn first_node(&self) -> &Node {
        self.inner.get(&0).unwrap()
    }
}

#[derive(Debug, Clone)]
struct DecisionTreeModel {
    num_features: u32,
    trees: Vec<Tree>,
    average_tree_output: bool,
    base_score: f64,
}

impl TryFrom<parser::UnknownNode> for Node {
    type Error = anyhow::Error;

    #[tracing::instrument(err)]
    fn try_from(value: parser::UnknownNode) -> Result<Self, Self::Error> {
        if let Some(true) = value.has_categorical_split {
            anyhow::bail!("Unsupported categorical split");
        }

        if let Some(false) = value.category_list_right_child {
            anyhow::bail!("Unsupported category list right child");
        }

        if let Some(node_type) = value.node_type.as_deref() {
            match node_type {
                "numerical_test_node" => Ok(Node::TestNode(TestNode {
                    node_id: value.node_id,
                    split_feature_id: value.split_feature_id.unwrap(),
                    comparison_op: value.comparison_op.as_deref().unwrap().to_string(),
                    threshold: value.threshold.unwrap(),
                    left_child: value.left_child.unwrap(),
                    right_child: value.right_child.unwrap(),
                })),

                "categorical_test_node" => Ok(Node::Categorical(CategoricalTestNode {
                    node_id: value.node_id,
                    left_child: value.left_child.unwrap(),
                    right_child: value.right_child.unwrap(),
                    split_feature_id: value.split_feature_id.unwrap(),
                    category_list: value.category_list.unwrap(),
                })),

                _ => anyhow::bail!("Not supported node type: {}", node_type),
            }

            // this one is probably used only for __builtin_expect
            //if let Some(false) = value.default_left {
            //    anyhow::bail!("Not supported default left");
            //}
        } else {
            Ok(Node::Leaf(Leaf {
                node_id: value.node_id,
                leaf_value: value.leaf_value.unwrap(),
            }))
        }
    }
}

impl TryFrom<&parser::Tree> for Tree {
    type Error = anyhow::Error;

    #[tracing::instrument(err, skip_all)]
    fn try_from(value: &parser::Tree) -> Result<Self, Self::Error> {
        let mut hash_map: HashMap<u32, Node> = HashMap::with_capacity(value.nodes.len());

        for tree in value.nodes.clone() {
            let node: Node = tree.try_into()?;

            hash_map.insert(node.id(), node);
        }
        Ok(Tree {
            inner: hash_map,
            has_categorical_split: value.has_categorical_split,
        })
    }
}

impl DecisionTreeModel {
    #[tracing::instrument(err)]
    fn from_json(path: &str) -> Result<Self> {
        tracing::info!("Parsing JSON from {} ...", path);
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::with_capacity(1024 * 1024, file);
        let json: parser::ModelJSON = serde_json::from_reader(reader)?;

        anyhow::ensure!(
            json.leaf_output_type == "float64" || json.leaf_output_type == "float32",
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
        let trees: Result<Vec<Tree>> = json.trees.iter().map(|i| i.try_into()).collect();
        let trees = trees?;

        anyhow::ensure!(
            json.base_scores.len() == 1,
            std::format!(
                "Unsupported json.base_scores.len() == {:?}",
                json.base_scores.len()
            )
        );

        Ok(DecisionTreeModel {
            num_features: json.num_feature,
            trees,
            average_tree_output: json.average_tree_output,
            base_score: *json.base_scores.get(0).unwrap(),
        })
    }
}

#[tracing::instrument(err)]
fn buf_writer(path: &str) -> Result<std::io::BufWriter<std::fs::File>> {
    let file = std::fs::File::create(path)?;
    Ok(std::io::BufWriter::with_capacity(1024 * 1024, file))
}

#[tracing::instrument(err, level = "trace")]
fn emit_node(
    node: &Node,
    nodes: &HashMap<u32, Node>,
    depth: u32,
    fp: &mut std::io::BufWriter<std::fs::File>,
) -> Result<()> {
    let indent = "  ".repeat(depth as usize);

    match node {
        Node::TestNode(TestNode {
            node_id: _,
            split_feature_id,
            comparison_op,
            threshold,
            left_child,
            right_child,
        }) => {
            fp.write_all(indent.as_bytes())?;
            fp.write_all(b"if (arr[")?;
            fp.write_all(split_feature_id.to_string().as_bytes())?;
            fp.write_all(b"] ")?;
            fp.write_all(comparison_op.as_bytes())?;
            fp.write_all(b" ")?;
            fp.write_all(threshold.to_string().as_bytes())?;
            fp.write_all(b")")?;
            fp.write_all(b" {\n")?;

            let left_child_node = nodes.get(left_child).unwrap();

            emit_node(left_child_node, nodes, depth + 1, fp)?;
            fp.write_all(indent.as_bytes())?;
            fp.write_all(b"} else {\n")?;

            let right_child_node = nodes.get(right_child).unwrap();

            emit_node(right_child_node, nodes, depth + 1, fp)?;
            fp.write_all(indent.as_bytes())?;
            fp.write_all(b"}\n")?;
        }
        Node::Leaf(Leaf {
            node_id: _,
            leaf_value,
        }) => {
            fp.write_all(indent.as_bytes())?;

            if *leaf_value < 0.0 {
                fp.write_all(b"result -= ")?;
                fp.write_all(leaf_value.abs().to_string().as_bytes())?;
            } else {
                fp.write_all(b"result += ")?;
                fp.write_all(leaf_value.to_string().as_bytes())?;
            }

            fp.write_all(b";\n")?;
            fp.flush()?;
        }

        Node::Categorical(CategoricalTestNode {
            node_id: _,
            left_child,
            right_child,
            split_feature_id,
            category_list,
        }) => {
            fp.write_all(indent.as_bytes())?;
            fp.write_all(b"tmp = (int)arr[")?;
            fp.write_all(split_feature_id.to_string().as_bytes())?;
            fp.write_all(b"];\n")?;

            fp.write_all(indent.as_bytes())?;
            fp.write_all(b"if (!List.of(")?;

            for (id, cat) in category_list.iter().enumerate() {
                fp.write_all(cat.to_string().as_bytes())?;
                if id != category_list.len() - 1 {
                    fp.write_all(b",")?
                }
            }

            fp.write_all(b").contains(tmp)) {\n")?;

            let left_child_node = nodes.get(left_child).unwrap();

            emit_node(left_child_node, nodes, depth + 1, fp)?;

            fp.write_all(indent.as_bytes())?;

            fp.write_all(b"} else {\n")?;

            let right_child_node = nodes.get(right_child).unwrap();

            emit_node(right_child_node, nodes, depth + 1, fp)?;

            //fp.write_all(std::format!("// {:?}", node).as_bytes())?;

            fp.write_all(indent.as_bytes())?;
            fp.write_all(b"}\n")?;
        }
    }

    Ok(())
}

#[tracing::instrument(err, skip(tree))]
fn emit_tree(tree: &Tree, destination: &str, tu_id: u32, package: &str) -> Result<()> {
    let file_name = std::format!("Tree{:}.java", tu_id);
    let destination = std::path::PathBuf::from(destination);
    let destination = destination.join(file_name);
    let destination = destination.to_str().unwrap();

    tracing::info!("Emitting tree into {:?} ...", destination);
    let mut fp = buf_writer(destination)?;

    fp.write_all(b"package ")?;
    fp.write_all(package.as_bytes())?;
    fp.write_all(b";\n\n")?;

    fp.write_all(b"import java.util.List;\n\n")?;

    fp.write_all(b"final class Tree")?;
    fp.write_all(tu_id.to_string().as_bytes())?;
    fp.write_all(b"{\n\n")?;

    fp.write_all(b"  public static double predict(double[] arr){\n")?;

    if tree.has_categorical_split {
        fp.write_all(b"    int tmp;\n")?;
    }

    fp.write_all(b"    double result = 0.0;\n")?;

    emit_node(tree.first_node(), &tree.inner, 2, &mut fp)?;
    fp.write_all(b"    return result;\n  }\n}")?;

    fp.flush()?;
    Ok(())
}

#[tracing::instrument(err)]
fn emit_ensemble(
    destination: &str,
    package: &str,
    num_features: u32,
    num_trees: u32,
    average_tree_output: bool,
    base_score: f64,
) -> Result<()> {
    let destination = std::path::PathBuf::from(destination);
    let destination = destination.join("TreeEnsemble.java");
    let destination = destination.to_str().unwrap();

    tracing::debug!("Emitting ensemble into {:?}...", destination);
    let mut fp = buf_writer(destination)?;

    fp.write_all(b"package ")?;
    fp.write_all(package.as_bytes())?;
    fp.write_all(b";\n\n")?;
    fp.write_all(b"public final class TreeEnsemble {\n\n")?;
    fp.write_all(b"  private TreeEnsemble(){ }\n")?;
    fp.write_all(b"  private static final int NUM_FEATURES = ")?;
    fp.write_all(num_features.to_string().as_bytes())?;
    fp.write_all(b";\n\n")?;

    fp.write_all(b"  public static double predict(double[] arr){\n")?;
    fp.write_all(b"     assert arr.length == NUM_FEATURES;\n")?;
    fp.write_all(b"     double result = 0.0;\n")?;

    for tu_id in 0..num_trees {
        fp.write_all(b"     result += Tree")?;
        fp.write_all(tu_id.to_string().as_bytes())?;
        fp.write_all(b".predict(arr);\n")?;
    }

    if average_tree_output {
        fp.write_all(b"     result /= ")?;
        fp.write_all(num_trees.to_string().as_bytes())?;
        fp.write_all(b".0;\n")?;
    }

    if base_score != 0.0 {
        if base_score > 0.0 {
            fp.write_all(b"     result += ")?;
            fp.write_all(base_score.to_string().as_bytes())?;
            fp.write_all(b";\n")?;
        } else {
            fp.write_all(b"     result -= ")?;
            fp.write_all(base_score.abs().to_string().as_bytes())?;
            fp.write_all(b";\n")?;
        }
    }

    fp.write_all(b"     return result;\n  }\n\n}")?;

    fp.flush()?;
    tracing::debug!("Emitting ensemble done.");
    Ok(())
}

#[tracing::instrument(skip(model), err)]
fn emit_java(
    model: &DecisionTreeModel,
    destination: &str,
    package: &str,
) -> Result<()> {
    tracing::info!("Emitting into {} ...", destination);

    let num_trees = model.trees.len() as u32;

    let java_dir = package.replace(".", "/");

    let destination = std::path::PathBuf::from(destination);
    let destination = destination.join(java_dir);
    let destination = destination.to_str().unwrap();

    tracing::info!("destination = {:?}", destination);

    emit_ensemble(
        destination,
        package,
        model.num_features,
        num_trees,
        model.average_tree_output,
        model.base_score,
    )?;

    let result: Result<Vec<()>> = model
        .trees
        .par_iter()
        .enumerate()
        .map(|(tu_id, tree)| emit_tree(tree, destination, tu_id as u32, package))
        .collect();

    result?;
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::try_parse()?;
    setup_tracing()?;

    let model = DecisionTreeModel::from_json(cli.source.as_str())?;

    emit_java(
        &model,
        cli.destination.as_str(),
        cli.package.as_str(),
    )
}
