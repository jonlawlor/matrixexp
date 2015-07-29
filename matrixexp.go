// Copyright 2015 Jonathan J Lawlor. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

// A work in progress, heavily influenced by gonum/matrix and my previous work
// in relational algebra.  Currently only implements matrices of float64.

package matrixexp

// Matrix represents any mathematical matrix, and defines its Algebra.
type Matrix interface {

	// not algebra, but very helpful
	Dims() (r, c int)        // matrix dimensions
	At(r, c int) float64     // get a value from a given row, column index
	Set(r, c int, v float64) // set a value at a given row, column index
	Vector() []float64       // vector returns all of the values in the matrix as a []float64, in row order

	// Matrix Algebra
	T() Matrix             // transpose
	Add(Matrix) Matrix     // matrix addition
	Sub(Matrix) Matrix     // matrix subtraction
	Mul(Matrix) Matrix     // matrix multiplication
	MulElem(Matrix) Matrix // element-wise multiplication
	DivElem(Matrix) Matrix // element-wise division
	Inv() Matrix           // matrix inversion
}
