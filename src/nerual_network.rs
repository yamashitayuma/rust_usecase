


#[test]
fn array_loops() {
    use ndarray::{ArrayD, Array1, Dimension, array, s, Dim};

    pub trait LayerBase<T> {
        fn forward(&mut self, x: &ArrayD<T>) -> ArrayD<T>;

        fn backward(&mut self, x: &ArrayD<T>, dx: &ArrayD<T>) -> ArrayD<T>;

        fn update(&mut self, lr: T);
    }

    pub trait MathFunc<T> {
        fn exp(&self) -> Self;
        fn log(&self, e: T) -> Self;
        fn log_natural(&self) -> Self;
    }



    // owned array to view 
    let a = array![1.0, 2.0, 3.0];
    let b: Array<f64, Dim<[usize; 1]>> = 1.0 * &a.slice(s![..2]);  
    // print!("{:?}", b);

    use ndarray::{Zip, Array, Array2, ArrayBase, OwnedRepr, Dim};

    let mut a: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> = Array::zeros((10, 10));
    
    let mut row = a.row_mut(0);

    row.fill(1.);
    
    let sum = row.sum();

    print!("sum = {:#?}", sum);

    // let mut b = Array::zeros(a.rows());

    // Zip::from(a.genrows())
    //     .and(&mut b)
    //     .apply(|a_row, b_elt| {
    //         *b_elt = a_row[a.cols() - 1] + a_row[0];
    //     });
    // println!("b {:?}", b);

    // let mut c = Array::zeros(10);
    // Zip::from(&mut c)
    //     .and(&b)
    //     .apply(|c_elt, &b_elt| {
    //         *c_elt = b_elt + 1.;
    //     });
    // println!("c {:?}", c);

    // use ndarray::Array2;
    // type M = Array2<f64>;
    // let mut a = M::zeros((12, 8));
    // let b = M::from_elem(a.dim(), 1.);
    // let c = M::from_elem(a.dim(), 2.);
    // let d = M::from_elem(a.dim(), 3.);


    // Zip::from(&mut a)
    //     .and(&b)
    //     .and(&c)
    //     .and(&d)
    //     .apply(|w, &x, &y, &z| {
    //         *w += x + y * z;
    //     });
    // println!("{:?}", a);
}