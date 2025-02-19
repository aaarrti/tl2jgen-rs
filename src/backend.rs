use crate::ir::{Leaf, Node, TestNode};

fn buf_writer(path: &std::path::Path) -> std::io::BufWriter<std::fs::File> {
    let file = std::fs::File::create(path).unwrap();
    std::io::BufWriter::with_capacity(1024 * 1024, file)
}

#[inline(always)]
fn node_matches(id: &u32, node: &Node) -> bool {
    match node {
        Node::Leaf(Leaf {
            node_id,
            leaf_value: _,
        }) => *node_id == *id,
        Node::TestNode(TestNode {
            node_id,
            split_feature_id: _,
            comparison_op: _,
            threshold: _,
            left_child: _,
            right_child: _,
        }) => *node_id == *id,
    }
}

pub mod c {

    use super::{buf_writer, node_matches};
    use crate::ir::{Leaf, ModelIR, Node, TestNode, Tree};
    use anyhow::Result;
    use indicatif::ParallelProgressIterator;
    use rayon::prelude::*;
    use std::io::Write;

    fn emit_header(path: &std::path::Path, num_trees: usize) -> Result<()> {
        tracing::debug!("Emitting header into {:?}...", path);
        let mut fp = buf_writer(path);

        fp.write("#define EXPORT __attribute__((visibility(\"default\")))\n\n".as_bytes())?;
        fp.write("EXPORT int get_num_features(void);\n".as_bytes())?;
        fp.write("EXPORT double predict(double* data);\n\n".as_bytes())?;

        for tu_id in 0..num_trees {
            fp.write("void predict_unit".as_bytes())?;
            fp.write(tu_id.to_string().as_bytes())?;
            fp.write("(double* data, double* result);\n".as_bytes())?;
        }

        fp.flush()?;
        Ok(())
    }

    fn emit_main(path: &std::path::Path, num_trees: usize, num_features: u32) -> Result<()> {
        tracing::debug!("Emitting main into {:?}...", path);
        let mut fp = buf_writer(path);
        fp.write("#include \"header.h\"\n\n".as_bytes())?;
        fp.write("EXPORT int get_num_features(void){\n".as_bytes())?;

        fp.write("  return ".as_bytes())?;
        fp.write(num_features.to_string().as_bytes())?;
        fp.write(";\n}\n".as_bytes())?;

        fp.write("EXPORT double predict(double* data){\n  double result = 0;\n".as_bytes())?;

        for tu_id in 0..num_trees {
            fp.write("  predict_unit".as_bytes())?;
            fp.write(tu_id.to_string().as_bytes())?;
            fp.write("(data, &result);\n".as_bytes())?;
        }
        fp.write("  result /= ".as_bytes())?;
        fp.write(num_trees.to_string().as_bytes())?;
        fp.write(".0;\n  return result;\n}".as_bytes())?;
        fp.flush()?;
        Ok(())
    }

    fn emit_node(
        nodes: &[Node],
        node: &Node,
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
                fp.write(indent.as_bytes())?;
                fp.write("if(data[".as_bytes())?;
                fp.write(split_feature_id.to_string().as_bytes())?;
                fp.write("] ".as_bytes())?;
                fp.write(comparison_op.as_bytes())?;
                fp.write(" (double)".as_bytes())?;
                fp.write(threshold.to_string().as_bytes())?;
                fp.write(")".as_bytes())?;
                fp.write("{\n".as_bytes())?;

                let left_child_node = nodes.iter().filter(|i| node_matches(left_child, i)).next();

                match left_child_node {
                    Some(left) => {
                        emit_node(nodes, left, depth + 1, fp)?;
                        fp.write(indent.as_bytes())?;
                        fp.write("} else {\n".as_bytes())?;
                    }
                    None => {
                        anyhow::bail!("No node found for {}", left_child);
                    }
                }

                let right_child_node = nodes.iter().filter(|i| node_matches(right_child, i)).next();

                match right_child_node {
                    Some(right) => {
                        emit_node(nodes, right, depth + 1, fp)?;
                        fp.write(indent.as_bytes())?;
                        fp.write("}\n".as_bytes())?;
                    }
                    None => {
                        anyhow::bail!("No node found for {}", left_child);
                    }
                }

                Ok(())
            }
            Node::Leaf(Leaf {
                node_id: _,
                leaf_value,
            }) => {
                fp.write(indent.as_bytes())?;
                fp.write("result[0] += ".as_bytes())?;
                fp.write(leaf_value.to_string().as_bytes())?;
                fp.write(";\n".as_bytes())?;
                fp.flush()?;

                Ok(())
            }
        }
    }

    fn emit_tree(tree: &Tree<Node>, path: &str, tu_id: u32) -> Result<()> {
        let path = std::path::Path::new(path);
        let tu_path = std::format!("tu{}.c", tu_id);
        let tu_path = path.join(tu_path);
        let mut fp = buf_writer(tu_path.as_path());

        fp.write("#include \"header.h\"\n\n".as_bytes())?;

        fp.write("void predict_unit".as_bytes())?;
        fp.write(tu_id.to_string().as_bytes())?;
        fp.write("(double* data, double* result){\n".as_bytes())?;

        emit_node(
            tree.nodes.as_slice(),
            tree.nodes.get(0).unwrap(),
            1,
            &mut fp,
        )?;
        fp.write("\n}".as_bytes())?;

        Ok(())
    }

    pub fn emit(model: &ModelIR, path: &str) -> Result<()> {
        if let Err(err) = std::fs::create_dir(path) {
            if err.kind() != std::io::ErrorKind::AlreadyExists {
                anyhow::bail!("Failed to created directory: {}", err);
            }
        }

        let heeader_path = std::format!("{}/header.h", path);
        let header_path = std::path::Path::new(heeader_path.as_str());

        let main_path = std::format!("{}/main.c", path);
        let main_path = std::path::Path::new(main_path.as_str());

        let num_trees = model.trees.len();

        emit_header(header_path, num_trees)?;
        emit_main(main_path, num_trees, model.num_features)?;

        let result: Result<Vec<()>> = model
            .trees
            .par_iter()
            .progress_count(num_trees as u64)
            .enumerate()
            .map(|(tu_id, tree)| emit_tree(tree, path, tu_id as u32))
            .collect();

        result?;
        Ok(())
    }
}

pub mod java {

    use std::io::Write;

    use crate::ir::{Leaf, ModelIR, Node, TestNode};
    use anyhow::Result;

    use super::node_matches;

    pub fn emit(_model: &ModelIR, _path: &str) -> Result<()> {
        anyhow::bail!("Not implemented")
    }
}
