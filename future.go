// Copyright 2015 Jonathan J Lawlor. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

package matrixexp

import (
	"github.com/gonum/blas/blas64"
)

// NewFuture constructs a Future MatrixLiteral from a Matrix Expression and
// then begins evaluating it.
func NewFuture(M MatrixExp) *Future {
	ch := make(chan struct{})
	r, c := M.Dims()
	F := &Future{
		r:  r,
		c:  c,
		ch: ch,
		m:  nil,
	}
	go func(M MatrixExp, F *Future, ch chan<- struct{}) {
		F.m = M.Eval()
		close(ch)
	}(M, F, ch)
	return F
}

// Future is a matrix literal that is asynchronously evaluating.
type Future struct {
	r, c int
	ch   <-chan struct{}
	m    MatrixLiteral
}

// Dims returns the matrix dimensions.
func (m1 *Future) Dims() (r, c int) {
	r = m1.r
	c = m1.c
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

// Copy creates a (deep) copy of the Matrix Expression.
func (m1 *Future) Copy() MatrixExp {
	//TODO(jonlawlor): handle the case where we want to copy a running job,
	// maybe with pub/sub?
	<-m1.ch

	return &Future{
		r:  m1.r,
		c:  m1.c,
		ch: m1.ch,
		m:  m1.m.Copy().Eval(),
	}
}

// Err returns the first error encountered while constructing the matrix expression.
func (m1 *Future) Err() error {

	// This is not ideal.  We can't tell if the matrix is invalid until we have
	// already calculated it, and by that time it is likely that it will have
	// panicked.  On the other hand, you should only get a future by Async, so
	// compiling the matrix expression that generates this future should yield
	// an error before here.
	<-m1.ch
	if err := m1.m.Err(); err != nil {
		return err
	}
	return nil
}

// T transposes a matrix.
func (m1 *Future) T() MatrixExp {
	return &T{m1}
}

// Add two matrices together.
func (m1 *Future) Add(m2 MatrixExp) MatrixExp {
	return &Add{
		Left:  m1,
		Right: m2,
	}
}

// Sub subtracts the right matrix from the left matrix.
func (m1 *Future) Sub(m2 MatrixExp) MatrixExp {
	return &Sub{
		Left:  m1,
		Right: m2,
	}
}

// Scale performs scalar multiplication.
func (m1 *Future) Scale(c float64) MatrixExp {
	return &Scale{
		C: c,
		M: m1,
	}
}

// Mul performs matrix multiplication.
func (m1 *Future) Mul(m2 MatrixExp) MatrixExp {
	return &Mul{
		Left:  m1,
		Right: m2,
	}
}

// MulElem performs element-wise multiplication.
func (m1 *Future) MulElem(m2 MatrixExp) MatrixExp {
	return &MulElem{
		Left:  m1,
		Right: m2,
	}
}

// DivElem performs element-wise division.
func (m1 *Future) DivElem(m2 MatrixExp) MatrixExp {
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
