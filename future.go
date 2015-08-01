// Copyright 2015 Jonathan J Lawlor. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

package matrixexp

import (
	"github.com/gonum/blas/blas64"
)

func NewFuture(M MatrixExpr) *Future {
	ch := make(chan struct{})
	F := &Future{
		ch: ch,
		m:  nil,
	}
	go func(M MatrixExpr, F *Future, ch chan<- struct{}) {
		F.m = M.Eval()
		close(ch)
	}(M, F, ch)
	return F
}

// Future is a matrix literal that is asynchronously evaluating.
type Future struct {
	ch <-chan struct{}
	m  MatrixLiteral
}

// Dims returns the matrix dimensions.
func (m1 *Future) Dims() (r, c int) {
	<-m1.ch
	r, c = m1.m.Dims()
	return
}

// At returns the value at a given row, column index.
func (m1 *Future) At(r, c int) float64 {
	<-m1.ch
	return m1.m.At(r, c)
}

// Set changes the value at a given row, column index.
func (m1 *Future) Set(r, c int, v float64) {
	<-m1.ch
	m1.m.Set(r, c, v)
}

// Eval returns a matrix literal.
func (m1 *Future) Eval() MatrixLiteral {
	return m1
}

// T transposes a matrix.
func (m1 *Future) T() MatrixExpr {
	return &T{m1}
}

// Add two matrices together.
func (m1 *Future) Add(m2 MatrixExpr) MatrixExpr {
	return &Add{
		Left:  m1,
		Right: m2,
	}
}

// Sub subtracts the right matrix from the left matrix.
func (m1 *Future) Sub(m2 MatrixExpr) MatrixExpr {
	return &Sub{
		Left:  m1,
		Right: m2,
	}
}

// Mul performs matrix multiplication.
func (m1 *Future) Mul(m2 MatrixExpr) MatrixExpr {
	return &Mul{
		Left:  m1,
		Right: m2,
	}
}

// MulElem performs element-wise multiplication.
func (m1 *Future) MulElem(m2 MatrixExpr) MatrixExpr {
	return &MulElem{
		Left:  m1,
		Right: m2,
	}
}

// DivElem performs element-wise division.
func (m1 *Future) DivElem(m2 MatrixExpr) MatrixExpr {
	return &DivElem{
		Left:  m1,
		Right: m2,
	}
}

// AsVector returns a copy of the values in the matrix as a []float64, in row order.
func (m1 *Future) AsVector() []float64 {
	<-m1.ch
	return m1.m.AsVector()
}

// AsGeneral returns the matrix as a blas64.General (not a copy!)
func (m1 *Future) AsGeneral() blas64.General {
	<-m1.ch
	return m1.m.AsGeneral()
}
