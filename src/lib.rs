use ndarray::prelude::*;

fn spin_state(num_spins: u32, chain_configuration: u32, spin_number: u32) -> i32 {
    if spin_number > num_spins {
        panic!("Spin number should be <= num_spins");
    }
    let spin_number = spin_number - 1;
    let mask = 1 << spin_number;
    let state = chain_configuration & mask;
    if state != 0 {
        1
    } else {
        -1
    }
}

fn inter_chain_contribution(num_spins: u32, ch1: u32, ch2: u32) -> f64 {
    let mut hij = 0.0;
    for i in 1..=num_spins {
        hij += spin_state(num_spins, ch1, i) as f64 * spin_state(num_spins, ch2, i) as f64;
    }
    hij
}

fn chain_contribution(num_spins: u32, chain_conf: u32) -> f64 {
    let mut hij = 0.0;
    for i in 1..=num_spins {
        hij += spin_state(num_spins, chain_conf, i) as f64
            * spin_state(num_spins, chain_conf, i % num_spins + 1) as f64;
    }
    hij
}

fn hamiltonian(n: u32, ch1: u32, ch2: u32, jnn: f64) -> f64 {
    let hij = 0.5 * jnn * chain_contribution(n, ch1)
        + 0.5 * jnn * chain_contribution(n, ch2)
        + jnn * inter_chain_contribution(n, ch1, ch2);
    hij
}

fn tm_element(n: u32, ch1: u32, ch2: u32, jnn: f64, temp: f64) -> f64 {
    (-hamiltonian(n, ch1, ch2, jnn) / temp).exp()
}

fn unflatten_idx(idx: usize, n: usize) -> (usize, usize) {
    (idx / n, idx % n)
}

pub fn tm_n_parr(n: u32, jnn: f64, temp: f64) -> Array2<f64> {
    let n_states = 2_usize.pow(n);

    let mut tm_flat = Array::range(0., (n_states * n_states) as f64, 1.);

    tm_flat.par_map_inplace(|el| {
        let (i, j) = unflatten_idx(*el as usize, n_states);
        *el = tm_element(n, i as u32 + 1, j as u32 + 1, jnn, temp);
    });

    tm_flat.into_shape((n_states, n_states)).unwrap()
}
