use bit_vec::BitVec;
use num_bigint::BigInt;
use num_bigint::ToBigInt;
use std::collections::HashMap;
use rayon::prelude::*;
use std::fs::File;
use std::io::Write;

#[derive(Hash, Eq, PartialEq, Debug)]
struct Node {
    alphas: BitVec,
    candidates: BitVec,
    symmetric_sums: Vec<u8>,
}

fn test(alphas: &BitVec, r: usize, powers: &Vec<Vec<u8>>, k: usize) -> bool {
    // For k x r matrices where r >= k, the square submatrices have already been tested
    if r >= k {
        return true;
    }
    let submatrix: Vec<&u8> = powers
        .into_par_iter()
        .enumerate()
        .filter(|&(i, _)| alphas.get(i).unwrap())
        .map(|(_, e)| e.par_iter().take(k).collect::<Vec<&u8>>())
        .flatten()
        .collect();
    // For r = 2 submatrices, only one square submatrix that includes row 0 and row k-1
    if r == 2 {
        let submatrix = submatrix
            .into_par_iter()
            .enumerate()
            .filter(|&(i, _)| i % k == 0 || i % k == k - 1)
            .map(|(_, e)| e)
            .cloned()
            .collect();
        return isa_l::gf_invert_matrix(submatrix).is_some();
    }
    // Otherwise, find all r sized subsets of the k rows that include row 0 and row k-1
    let mut subset: BigInt = (ToBigInt::to_bigint(&1).unwrap() << (r-2)) - 1;
    while subset < (ToBigInt::to_bigint(&1).unwrap() << (k-2)) {
        //Map subset to corresponding submatrix
        let submatrix = submatrix
            .clone()
            .into_par_iter()
            .enumerate()
            .filter(|&(i, _)| i % k == 0 || i % k == k - 1 || subset.bit(((i % k) - 1).try_into().unwrap()))
            .map(|(_, e)| e)
            .cloned()
            .collect();
        if isa_l::gf_invert_matrix(submatrix).is_none() {
            return false;
        }

        //Gosper's hack to generate next subset
        let c: BigInt = subset.clone() & (ToBigInt::to_bigint(&0).unwrap()-subset.clone());
        let r: BigInt = subset.clone() + c.clone();
        subset = (((r.clone() ^ subset.clone()) >> 2) / c.clone()) | r.clone();
    }
    true
}

fn extend_symmetric_sums(prev_sums: &Vec<u8>, new_alpha: u8) -> Vec<u8> {
    let mut new_sums = Vec::new();
    new_sums.push(prev_sums[0] ^ new_alpha);
    for i in 1..prev_sums.len() {
        new_sums.push(prev_sums[i] ^ isa_l::gf_mul(prev_sums[i-1], new_alpha));
    }
    new_sums.push(isa_l::gf_mul(prev_sums[prev_sums.len() - 1], new_alpha));
    new_sums
}

fn main() {
    let mut powers: Vec<Vec<u8>> = vec![vec![1; 256]; 256];
    for col in 0..=255 {
        let mut n: u8 = 1;
        for i in 0..256 {
            powers[col][i] = n;
            n = isa_l::gf_mul(n, col.try_into().unwrap());
        }
    }

    let k = 3;
    let mut cur_hashmap = HashMap::new();
    let mut cur_depth = 1;
    let alphas = BitVec::from_fn(256, |i| i == 1);
    let node = Node {
        alphas: alphas.clone(),
        candidates: BitVec::from_fn(256, |i| i > 1),
        symmetric_sums: vec![1],
    };
    cur_hashmap.insert(alphas, node);

    let mut w = File::create("results.txt").unwrap();
    loop {
        writeln!(&mut w, "Number of nodes at r = {}: {}", cur_depth, cur_hashmap.len()).unwrap();
        println!("Number of nodes at r = {}: {}", cur_depth, cur_hashmap.len());
        let mut votes_map = HashMap::new();
        // Nomination phase
        for (alphas, node) in cur_hashmap.iter() {
            for i in 0..=255 as u8 {
                if node.candidates.get(i.into()).unwrap() {
                    let mut new_alphas = alphas.clone();
                    new_alphas.set(i.into(), true);
                    let new_sums = extend_symmetric_sums(&node.symmetric_sums, i);
                    let mut new_candidates = new_alphas.clone();
                    new_candidates.negate();
                    let nominee_node = Node {
                        alphas: new_alphas,
                        candidates: new_candidates,
                        symmetric_sums: new_sums,
                    };
                    votes_map.entry(nominee_node).and_modify(|v| *v += 1).or_insert(1);
                }
            }
        }

        //Testing phase
        let mut new_map: HashMap<BitVec,Node> = votes_map.into_par_iter()
                                        .filter(|(node,v)| *v == cur_depth && test(&node.alphas, *v + 1, &powers, k)) //Number of votes is one less than the current depth (r) because the parent node constructed by removing "1" from the set of alphas does not exist in our map
                                        .map(|(node,_)| (node.alphas.clone(), node))
                                        .collect();
        //Handshake phase
        cur_hashmap
            .par_iter_mut()
            .for_each(|(alphas, node)| node.candidates = BitVec::from_fn(256, |i| {
                node.candidates.get(i).unwrap() && new_map.contains_key(&BitVec::from_fn(256,|j| j == i || alphas.get(j).unwrap()))
            }));

        new_map
            .par_iter_mut()
            .for_each(|(alphas, node)| {
                let mut new_candidates = node.candidates.clone();
                for (i,_) in alphas.iter().enumerate().filter(|(_,x)| *x).collect::<Vec<_>>() {
                    if i == 1 { continue } //only considering parents that contain 1
                    let mut parent = alphas.clone();
                    parent.set(i, false);
                    new_candidates.and(&cur_hashmap.get(&parent).unwrap().candidates);
                }
                if k > cur_depth {
                    new_candidates.set(node.symmetric_sums[0].into(), false);
                    for j in 2..cur_depth {
                        new_candidates.set(isa_l::gf_mul(node.symmetric_sums[j-1], isa_l::gf_inv(node.symmetric_sums[j-2])).into(), false);
                    }
                }
                node.candidates = new_candidates;
            });
        if new_map.len() == 0 {
            break;
        }
        cur_hashmap = new_map;
        cur_depth += 1;
    }
}
    
