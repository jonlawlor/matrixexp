// Copyright 2015 Jonathan J Lawlor. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

package rewrite

import (
	"github.com/jonlawlor/matrixexp"
	"reflect"
)

// A Compiler can transform matrix expressions by applying various optimizations
// to it, such as converting a.T().Mul(b.T()) to an appropriate call to DGEMM,
// or allowing concurrent evaluation of (a.Mul(b)).Add(c.Mul(d)) via Async.
type Compiler interface {
	// Compile transforms a matrix, or returns an error indicating a problem
	// with either the matrix expression or the compiler.
	Compile(matrixexp.MatrixExp) (matrixexp.MatrixExp, error)

	// MustCompile is just like Compile, except that instead of returning an error
	// it will panic if it encounters a problem.  It is intended for use in init()
	MustCompile(matrixexp.MatrixExp) matrixexp.MatrixExp
}

// Rewriter can convert one matrix expression to another.
type Rewriter interface {
	Rewrite(matrixexp.MatrixExp) (matrixexp.MatrixExp, error)
}

// template represents an example based rule for the compiler to follow.
type template struct {
	from reflect.Value
	to   reflect.Value
}

// Template produces a rewrite rule from a matrix expression template.
func Template(from, to matrixexp.MatrixExp) Rewriter {
	return &template{
		from: reflect.ValueOf(from),
		to:   reflect.ValueOf(to),
	}
}

// Rewrite applies a rewrite rule to a matrix expression.
func (r *template) Rewrite(m1 matrixexp.MatrixExp) (matrixexp.MatrixExp, error) {

	// Determine if the matrix expression matches the rewrite rule.
	// TODO(jonlawlor): implement some kind of reflection cache.
	m2 := m1.Copy()
	return m2, nil
}
