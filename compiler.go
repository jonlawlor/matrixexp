// Copyright 2015 Jonathan J Lawlor. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

package matrixexp

import (
	"reflect"
)

// A Compiler can transform matrix expressions by applying various optimizations
// to it, such as converting a.T().Mul(b.T()) to an appropriate call to DGEMM,
// or allowing concurrent evaluation of (a.Mul(b)).Add(c.Mul(d)) via Async.
type Compiler interface {
	// Compile transforms a matrix, or returns an error indicating a problem
	// with either the matrix expression or the compiler.
	Compile(MatrixExp) (MatrixExp, error)

	// MustCompile is just like Compile, except that instead of returning an error
	// it will panic if it encounters a problem.  It is intended for use in init()
	MustCompile(MatrixExp) MatrixExp
}

// Rewriter can convert one matrix expression to another.  It is intended for
// use with NewRewrite to construct compilers.
type Rewriter interface {
	Rewrite(MatrixExp) MatrixExp
}

// Rewrite represents a rule for a compiler to follow.
type rewrite struct {
	from reflect.Value
	to   reflect.Value
}

// NewRewrite produces a rewrite rule from a matrix expression template.
func NewRewrite(from, to MatrixExp) Rewriter {
	return &rewrite{
		from: follow(reflect.ValueOf(from)),
		to:   follow(reflect.ValueOf(to)),
	}
}

// Follow any pointers until we reach a concrete type or nil.
func follow(v reflect.Value) reflect.Value {
	for k := v.Kind(); k == reflect.Ptr; k = v.Kind() {
		v = v.Elem()
	}
	return v
}

// Rewrite applies a rewrite rule to a matrix expression.
func (r *rewrite) Rewrite(m1 MatrixExp) MatrixExp {

	// Determine if the matrix expression matches the rewrite rule.
	// TODO(jonlawlor): implement some kind of reflection cache.
	m2 := m1.Copy()
	r.apply(reflect.ValueOf(m2))
	return m2
}

func (r *rewrite) apply(v reflect.Value) {

}
