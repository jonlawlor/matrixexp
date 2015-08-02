// Copyright 2015 Jonathan J Lawlor. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

package matrixexp

import (
	"github.com/gonum/blas"
	"github.com/gonum/blas/blas64"
)

// Mul represents matrix multiplication.
type Mul struct {
	Left  MatrixExpr
	Right MatrixExpr
}

// Dims returns the matrix dimensions.
func (m1 *Mul) Dims() (r, c int) {
	r, _ = m1.Left.Dims()
	_, c = m1.Right.Dims()
	return
}

// At returns the value at a given row, column index.
func (m1 *Mul) At(r, c int) float64 {
	var v float64
	_, n := m1.Left.Dims()
	for i := 0; i < n; i++ {
		v += m1.Left.At(r, i) * m1.Right.At(i, c)
	}
	return v
}

// Eval returns a matrix literal.
func (m1 *Mul) Eval() MatrixLiteral {

	// This should be replaced with a call to Eval on each side, and then a type
	// switch to handle the various matrix literals.

	lm := m1.Left.Eval()
	rm := m1.Right.Eval()

	left := lm.AsGeneral()
	right := rm.AsGeneral()
	r, c := m1.Dims()
	m := blas64.General{
		Rows:   r,
		Cols:   c,
		Stride: c,
		Data:   make([]float64, r*c),
	}
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, left, right, 0, m)
	return &General{m}
}

// Copy creates a (deep) copy of the Matrix Expression.
func (m1 *Mul) Copy() MatrixExpr {
	return &Mul{
		Left:  m1.Left.Copy(),
		Right: m1.Right.Copy(),
	}
}

// T transposes a matrix.
func (m1 *Mul) T() MatrixExpr {
	return &T{m1}
}

// Add two matrices together.
func (m1 *Mul) Add(m2 MatrixExpr) MatrixExpr {
	return &Add{
		Left:  m1,
		Right: m2,
	}
}

// Sub subtracts the right matrix from the left matrix.
func (m1 *Mul) Sub(m2 MatrixExpr) MatrixExpr {
	return &Sub{
		Left:  m1,
		Right: m2,
	}
}

// Scale performs scalar multiplication.
func (m1 *Mul) Scale(c float64) MatrixExpr {
	return &Scale{
		C: c,
		M: m1,
	}
}

// Mul performs matrix multiplication.
func (m1 *Mul) Mul(m2 MatrixExpr) MatrixExpr {
	return &Mul{
		Left:  m1,
		Right: m2,
	}
}

// MulElem performs element-wise multiplication.
func (m1 *Mul) MulElem(m2 MatrixExpr) MatrixExpr {
	return &MulElem{
		Left:  m1,
		Right: m2,
	}
}

// DivElem performs element-wise division.
func (m1 *Mul) DivElem(m2 MatrixExpr) MatrixExpr {
	return &DivElem{
		Left:  m1,
		Right: m2,
	}
}
