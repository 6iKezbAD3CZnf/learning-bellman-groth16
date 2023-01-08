use ff::{Field, PrimeField};

mod dummy_engine;
use dummy_engine::*;

use std::ops::{AddAssign, MulAssign, SubAssign};

use bellman::groth16::{create_proof, generate_parameters};
use bellman::{Circuit, ConstraintSystem, SynthesisError};

struct CubeDemo<E: PrimeField> {
    x: Option<E>,
}

impl<E: PrimeField> Circuit<E> for CubeDemo<E> {
    fn synthesize<CS: ConstraintSystem<E>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        /*
           Flattened into quadratic equations (x^3 + x + 5 == 35):

           x         * x = sym_1
           sym_1     * x = y
           (y    +x) * 1 = sym_2
           (sym_2+5) * 1 = ~out
           1         * 0 = 0
           ~out      * 0 = 0

           Resulting R1CS with w = [~one, ~out, x, sym_1, y, sym_2]
        */

        let x = self.x;
        let x_var = cs.alloc(|| "x", || x.ok_or(SynthesisError::AssignmentMissing))?;

        let sym_1 = x.map(|e| e.square());
        let sym_1_var = cs.alloc(
            || "sym_1",
            || sym_1.ok_or(SynthesisError::AssignmentMissing),
        )?;

        cs.enforce(
            || "sym_1",
            |lc| lc + x_var,
            |lc| lc + x_var,
            |lc| lc + sym_1_var,
        );

        let y = sym_1.map(|mut e| {
            e.mul_assign(&x.unwrap());
            e
        });
        let y_var = cs.alloc(|| "y", || y.ok_or(SynthesisError::AssignmentMissing))?;

        cs.enforce(
            || "y",
            |lc| lc + sym_1_var,
            |lc| lc + x_var,
            |lc| lc + y_var,
        );

        let sym_2 = y.map(|mut e| {
            e.add_assign(&x.unwrap());
            e
        });
        let sym_2_var = cs.alloc(
            || "sym_2",
            || sym_2.ok_or(SynthesisError::AssignmentMissing),
        )?;

        cs.enforce(
            || "sym_2",
            |lc| lc + y_var + x_var,
            |lc| lc + CS::one(),
            |lc| lc + sym_2_var,
        );

        let out = sym_2.map(|mut e| {
            e.add_assign(&E::from_str_vartime("5").unwrap());
            e
        });
        let out_var = cs.alloc_input(|| "out", || out.ok_or(SynthesisError::AssignmentMissing))?;

        cs.enforce(
            || "out",
            |lc| lc + sym_2_var + (E::from_str_vartime("5").unwrap(), CS::one()),
            |lc| lc + CS::one(),
            |lc| lc + out_var,
        );

        Ok(())
    }
}

fn main() {
    let g1 = Fr::one();
    let g2 = Fr::one();
    let alpha = Fr::from(10);
    let beta = Fr::from(20);
    let gamma = Fr::from(30);
    let delta = Fr::from(40);
    let tau = Fr::from(50);

    let params = {
        let c = CubeDemo { x: None };

        generate_parameters::<DummyEngine, _>(c, g1, g2, alpha, beta, gamma, delta, tau).unwrap()
    };

    assert_eq!(7, params.h.len());

    let root_of_unity = Fr::root_of_unity().pow_vartime(&[1u64 << 7]);

    let mut points = Vec::with_capacity(8);
    for i in 0u64..8 {
        points.push(root_of_unity.pow_vartime(&[i]));
    }

    let delta_inverse = delta.invert().unwrap();
    let gamma_inverse = gamma.invert().unwrap();

    // [~one, ~out]
    assert_eq!(2, params.vk.ic.len());
    // [x, sym_1, y, sym_2]
    assert_eq!(4, params.l.len());

    assert_eq!(6, params.a.len());
    assert_eq!(2, params.b_g1.len());

    /*
    Lagrange interpolation polynomials in our evaluation domain:

    TODO: 自明な等式があるのはなぜ？Public Inputに対して生成されるように見える

    x         * x = sym_1
    sym_1     * x = y
    (y    +x) * 1 = sym_2
    (sym_2+5) * 1 = ~out
    1         * 0 = 0
    ~out      * 0 = 0

    ,-----------------------------------------------------------------------.
    |A TERM                                                                 |
    `-----------------------------------------------------------------------.
    | ~one  | 0     | 0     | 0     | 5     | 1     | 0     | 0     | 0     |
    | ~out  | 0     | 0     | 0     | 0     | 0     | 1     | 0     | 0     |
    | x     | 1     | 0     | 1     | 0     | 0     | 0     | 0     | 0     |
    | sym_1 | 0     | 1     | 0     | 0     | 0     | 0     | 0     | 0     |
    | y     | 0     | 0     | 1     | 0     | 0     | 0     | 0     | 0     |
    | sym_2 | 0     | 0     | 0     | 1     | 0     | 0     | 0     | 0     |
    `-------'-------'-------'-------'-------'-------'-------'-------'-------'
    ,-----------------------------------------------------------------------.
    |B TERM                                                                 |
    `-----------------------------------------------------------------------.
    | ~one  | 0     | 0     | 1     | 1     | 0     | 0     | 0     | 0     |
    | ~out  | 0     | 0     | 0     | 0     | 0     | 0     | 0     | 0     |
    | x     | 1     | 1     | 0     | 0     | 0     | 0     | 0     | 0     |
    | sym_1 | 0     | 0     | 0     | 0     | 0     | 0     | 0     | 0     |
    | y     | 0     | 0     | 0     | 0     | 0     | 0     | 0     | 0     |
    | sym_2 | 0     | 0     | 0     | 0     | 0     | 0     | 0     | 0     |
    `-------'-------'-------'-------'-------'-------'-------'-------'-------'
    ,-----------------------------------------------------------------------.
    |C TERM                                                                 |
    `-----------------------------------------------------------------------.
    | ~one  | 0     | 0     | 0     | 0     | 0     | 0     | 0     | 0     |
    | ~out  | 0     | 0     | 0     | 1     | 0     | 0     | 0     | 0     |
    | x     | 0     | 0     | 0     | 0     | 0     | 0     | 0     | 0     |
    | sym_1 | 1     | 0     | 0     | 0     | 0     | 0     | 0     | 0     |
    | y     | 0     | 1     | 0     | 0     | 0     | 0     | 0     | 0     |
    | sym_2 | 0     | 0     | 1     | 0     | 0     | 0     | 0     | 0     |
    `-------'-------'-------'-------'-------'-------'-------'-------'-------'

    ```sage
    r = 64513
    Fr = GF(r)
    omega = (Fr(5)^63)^(2^7)
    tau = Fr(50)
    R.<x> = PolynomialRing(Fr, 'x')
    u_0 = 42352*x^7 + 1895*x^6 + 44882*x^5 + 32256*x^4 + 38289*x^3 + 46490*x^2 + 35759*x + 16129
    u_1 = 5539*x^7 + 36716*x^6 + 6045*x^5 + 8064*x^4 + 58974*x^3 + 27797*x^2 + 58468*x + 56449
    u_2 = 28652*x^7 + 19733*x^5 + 48385*x^4 + 28652*x^3 + 19733*x + 48385
    u_3 = 58974*x^7 + 36716*x^6 + 58468*x^5 + 8064*x^4 + 5539*x^3 + 27797*x^2 + 6045*x + 56449
    u_4 = 36716*x^7 + 8064*x^6 + 27797*x^5 + 56449*x^4 + 36716*x^3 + 8064*x^2 + 27797*x + 56449
    u_5 = 58468*x^7 + 27797*x^6 + 58974*x^5 + 8064*x^4 + 6045*x^3 + 36716*x^2 + 5539*x + 56449

    v_0 = 30671*x^7 + 35861*x^6 + 22258*x^5 + 42761*x^3 + 44780*x^2 + 33336*x + 48385
    v_1 = 0
    v_2 = 50910*x^7 + 28652*x^6 + 50404*x^5 + 61988*x^3 + 19733*x^2 + 62494*x + 48385
    v_3 = 0
    v_4 = 0
    v_5 = 0

    w_0 = 0
    w_1 = 58468*x^7 + 27797*x^6 + 58974*x^5 + 8064*x^4 + 6045*x^3 + 36716*x^2 + 5539*x + 56449
    w_2 = 0
    w_3 = 56449*x^7 + 56449*x^6 + 56449*x^5 + 56449*x^4 + 56449*x^3 + 56449*x^2 + 56449*x + 56449
    w_4 = 58974*x^7 + 36716*x^6 + 58468*x^5 + 8064*x^4 + 5539*x^3 + 27797*x^2 + 6045*x + 56449
    w_5 = 36716*x^7 + 8064*x^6 + 27797*x^5 + 56449*x^4 + 36716*x^3 + 8064*x^2 + 27797*x + 56449

    t = (x - 1)*(x - 20201)*(x - 35676)*(x - 16153)*(x - 64512)*(x - 44312)*(x - 28837)*(x - 38360)
    q = (u_0 + 35*u_1 + 3*u_2 + 9*u_3 + 27*u_4 + 30*u_5) * (v_0 + 3*v_2) - (35*w_1 + 9*w_3 + 27*w_4 + 30*w_5)
    h = q/t

    h
    ```
    > 37601*x^6 + 56707*x^5 + 25690*x^4 + 55189*x^3 + 20983*x^2 + 64096*x + 32255
    */

    let u_i = [24975, 51376, 62695, 6266, 51231, 35786]
        .iter()
        .map(|e| Fr::from(*e))
        .collect::<Vec<Fr>>();
    let v_i = [22504, 0, 17730, 0, 0, 0]
        .iter()
        .map(|e| Fr::from(*e))
        .collect::<Vec<Fr>>();
    let w_i = [0, 35786, 0, 11464, 6266, 51231]
        .iter()
        .map(|e| Fr::from(*e))
        .collect::<Vec<Fr>>();

    for (u, a) in u_i.iter().zip(&params.a[..]) {
        assert_eq!(u, a);
    }
    for (v, b) in v_i
        .iter()
        .filter(|&&e| e != Fr::zero())
        .zip(&params.b_g1[..])
    {
        assert_eq!(v, b);
    }
    for (v, b) in v_i
        .iter()
        .filter(|&&e| e != Fr::zero())
        .zip(&params.b_g2[..])
    {
        assert_eq!(v, b);
    }
    for i in 0..5 {
        let mut tmp1 = beta;
        tmp1.mul_assign(&u_i[i]);

        let mut tmp2 = alpha;
        tmp2.mul_assign(&v_i[i]);

        tmp1.add_assign(&tmp2);
        tmp1.add_assign(&w_i[i]);

        if i < 2 {
            tmp1.mul_assign(&gamma_inverse);
            assert_eq!(tmp1, params.vk.ic[i]);
        } else {
            tmp1.mul_assign(&delta_inverse);
            assert_eq!(tmp1, params.l[i - 2]);
        }
    }

    // Check consistency of the other elements
    assert_eq!(alpha, params.vk.alpha_g1);
    assert_eq!(beta, params.vk.beta_g1);
    assert_eq!(beta, params.vk.beta_g2);
    assert_eq!(gamma, params.vk.gamma_g2);
    assert_eq!(delta, params.vk.delta_g1);
    assert_eq!(delta, params.vk.delta_g2);

    // TODO: check pvk
    // let pvk = prepare_verifying_key(&params.vk);

    let r = Fr::from(27134);
    let s = Fr::from(17146);

    let proof = {
        let c = CubeDemo {
            x: Fr::from_str_vartime("3"),
        };

        create_proof(c, &params, r, s).unwrap()
    };

    {
        // proof A = alpha + A(tau) + delta * r
        let mut expected_a = delta;
        expected_a.mul_assign(&r);
        expected_a.add_assign(&alpha);
        expected_a.add_assign(&u_i[0]);
        expected_a.add_assign({
            let mut tmp = u_i[1].clone();
            tmp.mul_assign(Fr::from(35));
            tmp
        });
        expected_a.add_assign({
            let mut tmp = u_i[2].clone();
            tmp.mul_assign(Fr::from(3));
            tmp
        });
        expected_a.add_assign({
            let mut tmp = u_i[3].clone();
            tmp.mul_assign(Fr::from(9));
            tmp
        });
        expected_a.add_assign({
            let mut tmp = u_i[4].clone();
            tmp.mul_assign(Fr::from(27));
            tmp
        });
        expected_a.add_assign({
            let mut tmp = u_i[5].clone();
            tmp.mul_assign(Fr::from(30));
            tmp
        });
        assert_eq!(proof.a, expected_a);
    }
    {
        // proof B = beta + B(tau) + delta * s
        let mut expected_b = delta;
        expected_b.mul_assign(&s);
        expected_b.add_assign(&beta);
        expected_b.add_assign(&v_i[0]);
        expected_b.add_assign({
            let mut tmp = v_i[1].clone();
            tmp.mul_assign(Fr::from(35));
            tmp
        });
        expected_b.add_assign({
            let mut tmp = v_i[2].clone();
            tmp.mul_assign(Fr::from(3));
            tmp
        });
        expected_b.add_assign({
            let mut tmp = v_i[3].clone();
            tmp.mul_assign(Fr::from(9));
            tmp
        });
        expected_b.add_assign({
            let mut tmp = v_i[4].clone();
            tmp.mul_assign(Fr::from(27));
            tmp
        });
        expected_b.add_assign({
            let mut tmp = v_i[5].clone();
            tmp.mul_assign(Fr::from(30));
            tmp
        });
        assert_eq!(proof.b, expected_b);
    }
    {
        let mut expected_c = Fr::zero();

        // A * s
        let mut tmp = proof.a;
        tmp.mul_assign(&s);
        expected_c.add_assign(&tmp);

        // B * r
        let mut tmp = proof.b;
        tmp.mul_assign(&r);
        expected_c.add_assign(&tmp);

        // delta * r * s
        let mut tmp = delta;
        tmp.mul_assign(&r);
        tmp.mul_assign(&s);
        expected_c.sub_assign(&tmp);

        // L query answer
        for (i, coeff) in [3, 9, 27, 30].iter().enumerate() {
            let coeff = Fr::from(*coeff);

            let mut tmp = params.l[i];
            tmp.mul_assign(&coeff);
            expected_c.add_assign(&tmp);
        }

        // H query answer
        for (i, coeff) in [32255, 64096, 20983, 55189, 25690, 56707, 37601]
            .iter()
            .enumerate()
        {
            let coeff = Fr::from(*coeff);

            let mut tmp = params.h[i];
            tmp.mul_assign(&coeff);
            expected_c.add_assign(&tmp);
        }

        assert_eq!(expected_c, proof.c);
    }

    println!("All tests passed!");
}
