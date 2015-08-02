// Copyright 2015 Jonathan J Lawlor. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

package matrixexp

// Async represents a matrix expression that evaluates into a Future.
type Async struct {
	M MatrixExpr
}

// Dims returns the matrix dimensions.
func (m1 *Async) Dims() (r, c int) {
	r, c = m1.M.Dims()
	return
}

// At returns the value at a given row, column index.
func (m1 *Async) At(r, c int) float64 {
	return m1.M.At(r, c)
}

// Eval returns a matrix literal.
func (m1 *Async) Eval() MatrixLiteral {
	return NewFuture(m1.M)
}

// T transposes a matrix.
func (m1 *Async) T() MatrixExpr {
	return m1.M
}

// Add two matrices together.
func (m1 *Async) Add(m2 MatrixExpr) MatrixExpr {
	return &Add{
		Left:  m1,
		Right: m2,
	}
}

// Sub subtracts the right matrix from the left matrix.
func (m1 *Async) Sub(m2 MatrixExpr) MatrixExpr {
	return &Sub{
		Left:  m1,
		Right: m2,
	}
}

// Mul performs matrix multiplication.
func (m1 *Async) Mul(m2 MatrixExpr) MatrixExpr {
	return &Mul{
		Left:  m1,
		Right: m2,
	}
}

// MulElem performs element-wise multiplication.
func (m1 *Async) MulElem(m2 MatrixExpr) MatrixExpr {
	return &MulElem{
		Left:  m1,
		Right: m2,
	}
}

// DivElem performs element-wise division.
func (m1 *Async) DivElem(m2 MatrixExpr) MatrixExpr {
	return &DivElem{
		Left:  m1,
		Right: m2,
	}
}
