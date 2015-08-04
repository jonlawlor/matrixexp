// Copyright 2015 Jonathan J Lawlor. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

// A work in progress, heavily influenced by gonum/matrix and my previous work
// in relational algebra.  Currently only implements matrices of float64.

package matrixexp

import (
	"github.com/gonum/blas/blas64"
)

// MatrixExp represents any mathematical matrix expression, and defines its Algebra.
type MatrixExp interface {

	// not a part of the algebra, but very helpful
	Dims() (r, c int)    // matrix dimensions
	At(r, c int) float64 // get a value from a given row, column index

	Eval() MatrixLiteral // Evaluates the matrix expression, producing a Matrix literal.
	Copy() MatrixExp     // creates a (deep) copy of the matrix expression

	// Originally Set was also a member of the Matrix method set, but then what
	// happens when you set (for example) a value in an Add Expression?  It is
	// clear that Set does not apply to all matrices, only to the ones with a
	// literal representation.

	// Matrix Algebra
	T() MatrixExp                // transpose
	Add(MatrixExp) MatrixExp     // matrix addition
	Sub(MatrixExp) MatrixExp     // matrix subtraction
	Scale(float64) MatrixExp     // scalar multiplication
	Mul(MatrixExp) MatrixExp     // matrix multiplication
	MulElem(MatrixExp) MatrixExp // element-wise multiplication
	DivElem(MatrixExp) MatrixExp // element-wise division
	// Inv() MatrixExp           // matrix inversion
}

// MatrixLiteral is a literal matrix, which can be converted to a blas64.General.
type MatrixLiteral interface {
	MatrixExp

	AsVector() []float64       // vector returns all of the values in the matrix as a []float64, in row order
	AsGeneral() blas64.General // returns a Matrix as a MatrixExp.General
	Set(r, c int, v float64)   // set a specific row, column to value
}

// Equals determines if two matrices are equal.
func Equals(m1, m2 MatrixExp) bool {
	r1, c1 := m1.Dims()
	r2, c2 := m2.Dims()

	if r1 != r2 || c1 != c2 {
		return false
	}
	mv1 := m1.Eval()
	mv2 := m2.Eval()

	v1 := mv1.AsVector()
	v2 := mv2.AsVector()
	for i, v := range v1 {
		if v2[i] != v {
			return false
		}
	}
	return true

}
