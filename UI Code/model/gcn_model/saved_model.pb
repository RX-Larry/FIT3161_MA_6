??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
l
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2		"
adj_xbool( "
adj_ybool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??
?
graph_convolution/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:od*)
shared_namegraph_convolution/kernel
?
,graph_convolution/kernel/Read/ReadVariableOpReadVariableOpgraph_convolution/kernel*
_output_shapes

:od*
dtype0
?
graph_convolution/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*'
shared_namegraph_convolution/bias
}
*graph_convolution/bias/Read/ReadVariableOpReadVariableOpgraph_convolution/bias*
_output_shapes
:d*
dtype0
?
graph_convolution_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*+
shared_namegraph_convolution_1/kernel
?
.graph_convolution_1/kernel/Read/ReadVariableOpReadVariableOpgraph_convolution_1/kernel*
_output_shapes

:dd*
dtype0
?
graph_convolution_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*)
shared_namegraph_convolution_1/bias
?
,graph_convolution_1/bias/Read/ReadVariableOpReadVariableOpgraph_convolution_1/bias*
_output_shapes
:d*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d@*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:d@*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:@*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:@ *
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
: *
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

: *
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/graph_convolution/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:od*0
shared_name!Adam/graph_convolution/kernel/m
?
3Adam/graph_convolution/kernel/m/Read/ReadVariableOpReadVariableOpAdam/graph_convolution/kernel/m*
_output_shapes

:od*
dtype0
?
Adam/graph_convolution/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*.
shared_nameAdam/graph_convolution/bias/m
?
1Adam/graph_convolution/bias/m/Read/ReadVariableOpReadVariableOpAdam/graph_convolution/bias/m*
_output_shapes
:d*
dtype0
?
!Adam/graph_convolution_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*2
shared_name#!Adam/graph_convolution_1/kernel/m
?
5Adam/graph_convolution_1/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/graph_convolution_1/kernel/m*
_output_shapes

:dd*
dtype0
?
Adam/graph_convolution_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*0
shared_name!Adam/graph_convolution_1/bias/m
?
3Adam/graph_convolution_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/graph_convolution_1/bias/m*
_output_shapes
:d*
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d@*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:d@*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:@ *
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

: *
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_3/kernel/m

)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/m
w
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes
:*
dtype0
?
Adam/graph_convolution/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:od*0
shared_name!Adam/graph_convolution/kernel/v
?
3Adam/graph_convolution/kernel/v/Read/ReadVariableOpReadVariableOpAdam/graph_convolution/kernel/v*
_output_shapes

:od*
dtype0
?
Adam/graph_convolution/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*.
shared_nameAdam/graph_convolution/bias/v
?
1Adam/graph_convolution/bias/v/Read/ReadVariableOpReadVariableOpAdam/graph_convolution/bias/v*
_output_shapes
:d*
dtype0
?
!Adam/graph_convolution_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*2
shared_name#!Adam/graph_convolution_1/kernel/v
?
5Adam/graph_convolution_1/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/graph_convolution_1/kernel/v*
_output_shapes

:dd*
dtype0
?
Adam/graph_convolution_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*0
shared_name!Adam/graph_convolution_1/bias/v
?
3Adam/graph_convolution_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/graph_convolution_1/bias/v*
_output_shapes
:d*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d@*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:d@*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:@ *
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

: *
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_3/kernel/v

)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/v
w
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?J
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?I
value?IB?I B?I
?
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-2

layer-9
layer_with_weights-3
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
R
trainable_variables
regularization_losses
	variables
	keras_api
 
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
 	variables
!	keras_api
h

"kernel
#bias
$trainable_variables
%regularization_losses
&	variables
'	keras_api
 
R
(trainable_variables
)regularization_losses
*	variables
+	keras_api
R
,trainable_variables
-regularization_losses
.	variables
/	keras_api
h

0kernel
1bias
2trainable_variables
3regularization_losses
4	variables
5	keras_api
h

6kernel
7bias
8trainable_variables
9regularization_losses
:	variables
;	keras_api
h

<kernel
=bias
>trainable_variables
?regularization_losses
@	variables
A	keras_api
h

Bkernel
Cbias
Dtrainable_variables
Eregularization_losses
F	variables
G	keras_api
?
Hiter

Ibeta_1

Jbeta_2
	Kdecay
Llearning_ratem?m?"m?#m?0m?1m?6m?7m?<m?=m?Bm?Cm?v?v?"v?#v?0v?1v?6v?7v?<v?=v?Bv?Cv?
V
0
1
"2
#3
04
15
66
77
<8
=9
B10
C11
 
V
0
1
"2
#3
04
15
66
77
<8
=9
B10
C11
?
Mnon_trainable_variables

Nlayers
Olayer_metrics
trainable_variables
Pmetrics
regularization_losses
	variables
Qlayer_regularization_losses
 
 
 
 
?
Rnon_trainable_variables
Slayer_metrics

Tlayers
Umetrics
trainable_variables
regularization_losses
	variables
Vlayer_regularization_losses
db
VARIABLE_VALUEgraph_convolution/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEgraph_convolution/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
Wnon_trainable_variables
Xlayer_metrics

Ylayers
Zmetrics
trainable_variables
regularization_losses
	variables
[layer_regularization_losses
 
 
 
?
\non_trainable_variables
]layer_metrics

^layers
_metrics
trainable_variables
regularization_losses
 	variables
`layer_regularization_losses
fd
VARIABLE_VALUEgraph_convolution_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEgraph_convolution_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1
 

"0
#1
?
anon_trainable_variables
blayer_metrics

clayers
dmetrics
$trainable_variables
%regularization_losses
&	variables
elayer_regularization_losses
 
 
 
?
fnon_trainable_variables
glayer_metrics

hlayers
imetrics
(trainable_variables
)regularization_losses
*	variables
jlayer_regularization_losses
 
 
 
?
knon_trainable_variables
llayer_metrics

mlayers
nmetrics
,trainable_variables
-regularization_losses
.	variables
olayer_regularization_losses
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

00
11
 

00
11
?
pnon_trainable_variables
qlayer_metrics

rlayers
smetrics
2trainable_variables
3regularization_losses
4	variables
tlayer_regularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

60
71
 

60
71
?
unon_trainable_variables
vlayer_metrics

wlayers
xmetrics
8trainable_variables
9regularization_losses
:	variables
ylayer_regularization_losses
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

<0
=1
 

<0
=1
?
znon_trainable_variables
{layer_metrics

|layers
}metrics
>trainable_variables
?regularization_losses
@	variables
~layer_regularization_losses
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

B0
C1
 

B0
C1
?
non_trainable_variables
?layer_metrics
?layers
?metrics
Dtrainable_variables
Eregularization_losses
F	variables
 ?layer_regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
^
0
1
2
3
4
5
6
7
	8

9
10
11
12
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
??
VARIABLE_VALUEAdam/graph_convolution/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/graph_convolution/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/graph_convolution_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/graph_convolution_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/graph_convolution/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/graph_convolution/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/graph_convolution_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/graph_convolution_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*4
_output_shapes"
 :??????????????????o*
dtype0*)
shape :??????????????????o
?
serving_default_input_2Placeholder*0
_output_shapes
:??????????????????*
dtype0
*%
shape:??????????????????
?
serving_default_input_3Placeholder*=
_output_shapes+
):'???????????????????????????*
dtype0*2
shape):'???????????????????????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2serving_default_input_3graph_convolution/kernelgraph_convolution/biasgraph_convolution_1/kernelgraph_convolution_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_5141
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename,graph_convolution/kernel/Read/ReadVariableOp*graph_convolution/bias/Read/ReadVariableOp.graph_convolution_1/kernel/Read/ReadVariableOp,graph_convolution_1/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp3Adam/graph_convolution/kernel/m/Read/ReadVariableOp1Adam/graph_convolution/bias/m/Read/ReadVariableOp5Adam/graph_convolution_1/kernel/m/Read/ReadVariableOp3Adam/graph_convolution_1/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp3Adam/graph_convolution/kernel/v/Read/ReadVariableOp1Adam/graph_convolution/bias/v/Read/ReadVariableOp5Adam/graph_convolution_1/kernel/v/Read/ReadVariableOp3Adam/graph_convolution_1/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *&
f!R
__inference__traced_save_5845
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamegraph_convolution/kernelgraph_convolution/biasgraph_convolution_1/kernelgraph_convolution_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/graph_convolution/kernel/mAdam/graph_convolution/bias/m!Adam/graph_convolution_1/kernel/mAdam/graph_convolution_1/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/mAdam/graph_convolution/kernel/vAdam/graph_convolution/bias/v!Adam/graph_convolution_1/kernel/vAdam/graph_convolution_1/bias/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/v*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_restore_5990??
?
b
C__inference_dropout_1_layer_call_and_return_conditional_losses_4852

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const?
dropout/MulMulinputsdropout/Const:output:0*
T0*4
_output_shapes"
 :??????????????????d2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*4
_output_shapes"
 :??????????????????d*
dtype0*
seed?'2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????d2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????d2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*4
_output_shapes"
 :??????????????????d2
dropout/Mul_1r
IdentityIdentitydropout/Mul_1:z:0*
T0*4
_output_shapes"
 :??????????????????d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????????????d:\ X
4
_output_shapes"
 :??????????????????d
 
_user_specified_nameinputs
?
x
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_4663

inputs
mask

identityt
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
strided_sliced
CastCastmask*

DstT0*

SrcT0
*0
_output_shapes
:??????????????????2
Castb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsCast:y:0ExpandDims/dim:output:0*
T0*4
_output_shapes"
 :??????????????????2

ExpandDimsm
mulMulinputsExpandDims:output:0*
T0*4
_output_shapes"
 :??????????????????d2
mulp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesl
SumSummul:z:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????d2
Sumt
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_1/reduction_indices~
Sum_1SumExpandDims:output:0 Sum_1/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
Sum_1m
truedivRealDivSum:output:0Sum_1:output:0*
T0*'
_output_shapes
:?????????d2	
truediv_
IdentityIdentitytruediv:z:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:??????????????????d:??????????????????:\ X
4
_output_shapes"
 :??????????????????d
 
_user_specified_nameinputs:VR
0
_output_shapes
:??????????????????

_user_specified_namemask
?

?
A__inference_dense_2_layer_call_and_return_conditional_losses_4718

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
]
7__inference_global_average_pooling1d_layer_call_fn_5594

inputs
mask

identity?
PartitionedCallPartitionedCallinputsmask*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_46632
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:??????????????????d:??????????????????:\ X
4
_output_shapes"
 :??????????????????d
 
_user_specified_nameinputs:VR
0
_output_shapes
:??????????????????

_user_specified_namemask
?
?
M__inference_graph_convolution_1_layer_call_and_return_conditional_losses_5549
inputs_0
inputs_11
shape_2_readvariableop_resource:dd)
add_readvariableop_resource:d
identity??add/ReadVariableOp?transpose/ReadVariableOpt
MatMulBatchMatMulV2inputs_1inputs_0*
T0*4
_output_shapes"
 :??????????????????d2
MatMulM
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:2
ShapeQ
Shape_1ShapeMatMul:output:0*
T0*
_output_shapes
:2	
Shape_1^
unstackUnpackShape_1:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack?
Shape_2/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

:dd*
dtype02
Shape_2/ReadVariableOpc
Shape_2Const*
_output_shapes
:*
dtype0*
valueB"d   d   2	
Shape_2`
	unstack_1UnpackShape_2:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2
Reshape/shapex
ReshapeReshapeMatMul:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????d2	
Reshape?
transpose/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

:dd*
dtype02
transpose/ReadVariableOpq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm?
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:dd2
	transposes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"d   ????2
Reshape_1/shapes
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:dd2
	Reshape_1v
MatMul_1MatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:?????????d2

MatMul_1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d2
Reshape_2/shape/2?
Reshape_2/shapePackunstack:output:0unstack:output:1Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape?
	Reshape_2ReshapeMatMul_1:product:0Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :??????????????????d2
	Reshape_2?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:d*
dtype02
add/ReadVariableOp?
addAddV2Reshape_2:output:0add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????d2
add\
ReluReluadd:z:0*
T0*4
_output_shapes"
 :??????????????????d2
Reluz
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :??????????????????d2

Identity~
NoOpNoOp^add/ReadVariableOp^transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:??????????????????d:'???????????????????????????: : 2(
add/ReadVariableOpadd/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp:^ Z
4
_output_shapes"
 :??????????????????d
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/1
?
?
A__inference_dense_3_layer_call_and_return_conditional_losses_4735

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
S
7__inference_global_average_pooling1d_layer_call_fn_5588

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_45332
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
b
C__inference_dropout_1_layer_call_and_return_conditional_losses_5508

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const?
dropout/MulMulinputsdropout/Const:output:0*
T0*4
_output_shapes"
 :??????????????????d2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*4
_output_shapes"
 :??????????????????d*
dtype0*
seed?'2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????d2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????d2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*4
_output_shapes"
 :??????????????????d2
dropout/Mul_1r
IdentityIdentitydropout/Mul_1:z:0*
T0*4
_output_shapes"
 :??????????????????d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????????????d:\ X
4
_output_shapes"
 :??????????????????d
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_4769
input_1
input_2

input_3
unknown:od
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:d@
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7: 
	unknown_8:
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_47422
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes
}:??????????????????o:??????????????????:'???????????????????????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :??????????????????o
!
_user_specified_name	input_1:YU
0
_output_shapes
:??????????????????
!
_user_specified_name	input_2:fb
=
_output_shapes+
):'???????????????????????????
!
_user_specified_name	input_3
?
`
A__inference_dropout_layer_call_and_return_conditional_losses_5440

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const?
dropout/MulMulinputsdropout/Const:output:0*
T0*4
_output_shapes"
 :??????????????????o2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*4
_output_shapes"
 :??????????????????o*
dtype0*
seed?'2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????o2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????o2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*4
_output_shapes"
 :??????????????????o2
dropout/Mul_1r
IdentityIdentitydropout/Mul_1:z:0*
T0*4
_output_shapes"
 :??????????????????o2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????????????o:\ X
4
_output_shapes"
 :??????????????????o
 
_user_specified_nameinputs
?
]
A__inference_flatten_layer_call_and_return_conditional_losses_4671

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????d   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????d2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?	
?
2__inference_graph_convolution_1_layer_call_fn_5559
inputs_0
inputs_1
unknown:dd
	unknown_0:d
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_graph_convolution_1_layer_call_and_return_conditional_losses_46402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????d2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:??????????????????d:'???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????d
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/1
?
?
$__inference_model_layer_call_fn_5392
inputs_0
inputs_1

inputs_2
unknown:od
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:d@
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7: 
	unknown_8:
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_47422
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes
}:??????????????????o:??????????????????:'???????????????????????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????o
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:??????????????????
"
_user_specified_name
inputs/1:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/2
?
?
$__inference_dense_layer_call_fn_5625

inputs
unknown:d@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_46842
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
??
?	
?__inference_model_layer_call_and_return_conditional_losses_5244
inputs_0
inputs_1

inputs_2C
1graph_convolution_shape_2_readvariableop_resource:od;
-graph_convolution_add_readvariableop_resource:dE
3graph_convolution_1_shape_2_readvariableop_resource:dd=
/graph_convolution_1_add_readvariableop_resource:d6
$dense_matmul_readvariableop_resource:d@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@ 5
'dense_1_biasadd_readvariableop_resource: 8
&dense_2_matmul_readvariableop_resource: 5
'dense_2_biasadd_readvariableop_resource:8
&dense_3_matmul_readvariableop_resource:5
'dense_3_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?$graph_convolution/add/ReadVariableOp?*graph_convolution/transpose/ReadVariableOp?&graph_convolution_1/add/ReadVariableOp?,graph_convolution_1/transpose/ReadVariableOpy
dropout/IdentityIdentityinputs_0*
T0*4
_output_shapes"
 :??????????????????o2
dropout/Identity?
graph_convolution/MatMulBatchMatMulV2inputs_2dropout/Identity:output:0*
T0*4
_output_shapes"
 :??????????????????o2
graph_convolution/MatMul?
graph_convolution/ShapeShape!graph_convolution/MatMul:output:0*
T0*
_output_shapes
:2
graph_convolution/Shape?
graph_convolution/Shape_1Shape!graph_convolution/MatMul:output:0*
T0*
_output_shapes
:2
graph_convolution/Shape_1?
graph_convolution/unstackUnpack"graph_convolution/Shape_1:output:0*
T0*
_output_shapes
: : : *	
num2
graph_convolution/unstack?
(graph_convolution/Shape_2/ReadVariableOpReadVariableOp1graph_convolution_shape_2_readvariableop_resource*
_output_shapes

:od*
dtype02*
(graph_convolution/Shape_2/ReadVariableOp?
graph_convolution/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"o   d   2
graph_convolution/Shape_2?
graph_convolution/unstack_1Unpack"graph_convolution/Shape_2:output:0*
T0*
_output_shapes
: : *	
num2
graph_convolution/unstack_1?
graph_convolution/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????o   2!
graph_convolution/Reshape/shape?
graph_convolution/ReshapeReshape!graph_convolution/MatMul:output:0(graph_convolution/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????o2
graph_convolution/Reshape?
*graph_convolution/transpose/ReadVariableOpReadVariableOp1graph_convolution_shape_2_readvariableop_resource*
_output_shapes

:od*
dtype02,
*graph_convolution/transpose/ReadVariableOp?
 graph_convolution/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2"
 graph_convolution/transpose/perm?
graph_convolution/transpose	Transpose2graph_convolution/transpose/ReadVariableOp:value:0)graph_convolution/transpose/perm:output:0*
T0*
_output_shapes

:od2
graph_convolution/transpose?
!graph_convolution/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"o   ????2#
!graph_convolution/Reshape_1/shape?
graph_convolution/Reshape_1Reshapegraph_convolution/transpose:y:0*graph_convolution/Reshape_1/shape:output:0*
T0*
_output_shapes

:od2
graph_convolution/Reshape_1?
graph_convolution/MatMul_1MatMul"graph_convolution/Reshape:output:0$graph_convolution/Reshape_1:output:0*
T0*'
_output_shapes
:?????????d2
graph_convolution/MatMul_1?
#graph_convolution/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d2%
#graph_convolution/Reshape_2/shape/2?
!graph_convolution/Reshape_2/shapePack"graph_convolution/unstack:output:0"graph_convolution/unstack:output:1,graph_convolution/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2#
!graph_convolution/Reshape_2/shape?
graph_convolution/Reshape_2Reshape$graph_convolution/MatMul_1:product:0*graph_convolution/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :??????????????????d2
graph_convolution/Reshape_2?
$graph_convolution/add/ReadVariableOpReadVariableOp-graph_convolution_add_readvariableop_resource*
_output_shapes
:d*
dtype02&
$graph_convolution/add/ReadVariableOp?
graph_convolution/addAddV2$graph_convolution/Reshape_2:output:0,graph_convolution/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????d2
graph_convolution/add?
graph_convolution/ReluRelugraph_convolution/add:z:0*
T0*4
_output_shapes"
 :??????????????????d2
graph_convolution/Relu?
dropout_1/IdentityIdentity$graph_convolution/Relu:activations:0*
T0*4
_output_shapes"
 :??????????????????d2
dropout_1/Identity?
graph_convolution_1/MatMulBatchMatMulV2inputs_2dropout_1/Identity:output:0*
T0*4
_output_shapes"
 :??????????????????d2
graph_convolution_1/MatMul?
graph_convolution_1/ShapeShape#graph_convolution_1/MatMul:output:0*
T0*
_output_shapes
:2
graph_convolution_1/Shape?
graph_convolution_1/Shape_1Shape#graph_convolution_1/MatMul:output:0*
T0*
_output_shapes
:2
graph_convolution_1/Shape_1?
graph_convolution_1/unstackUnpack$graph_convolution_1/Shape_1:output:0*
T0*
_output_shapes
: : : *	
num2
graph_convolution_1/unstack?
*graph_convolution_1/Shape_2/ReadVariableOpReadVariableOp3graph_convolution_1_shape_2_readvariableop_resource*
_output_shapes

:dd*
dtype02,
*graph_convolution_1/Shape_2/ReadVariableOp?
graph_convolution_1/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"d   d   2
graph_convolution_1/Shape_2?
graph_convolution_1/unstack_1Unpack$graph_convolution_1/Shape_2:output:0*
T0*
_output_shapes
: : *	
num2
graph_convolution_1/unstack_1?
!graph_convolution_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2#
!graph_convolution_1/Reshape/shape?
graph_convolution_1/ReshapeReshape#graph_convolution_1/MatMul:output:0*graph_convolution_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????d2
graph_convolution_1/Reshape?
,graph_convolution_1/transpose/ReadVariableOpReadVariableOp3graph_convolution_1_shape_2_readvariableop_resource*
_output_shapes

:dd*
dtype02.
,graph_convolution_1/transpose/ReadVariableOp?
"graph_convolution_1/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2$
"graph_convolution_1/transpose/perm?
graph_convolution_1/transpose	Transpose4graph_convolution_1/transpose/ReadVariableOp:value:0+graph_convolution_1/transpose/perm:output:0*
T0*
_output_shapes

:dd2
graph_convolution_1/transpose?
#graph_convolution_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"d   ????2%
#graph_convolution_1/Reshape_1/shape?
graph_convolution_1/Reshape_1Reshape!graph_convolution_1/transpose:y:0,graph_convolution_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:dd2
graph_convolution_1/Reshape_1?
graph_convolution_1/MatMul_1MatMul$graph_convolution_1/Reshape:output:0&graph_convolution_1/Reshape_1:output:0*
T0*'
_output_shapes
:?????????d2
graph_convolution_1/MatMul_1?
%graph_convolution_1/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d2'
%graph_convolution_1/Reshape_2/shape/2?
#graph_convolution_1/Reshape_2/shapePack$graph_convolution_1/unstack:output:0$graph_convolution_1/unstack:output:1.graph_convolution_1/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2%
#graph_convolution_1/Reshape_2/shape?
graph_convolution_1/Reshape_2Reshape&graph_convolution_1/MatMul_1:product:0,graph_convolution_1/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :??????????????????d2
graph_convolution_1/Reshape_2?
&graph_convolution_1/add/ReadVariableOpReadVariableOp/graph_convolution_1_add_readvariableop_resource*
_output_shapes
:d*
dtype02(
&graph_convolution_1/add/ReadVariableOp?
graph_convolution_1/addAddV2&graph_convolution_1/Reshape_2:output:0.graph_convolution_1/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????d2
graph_convolution_1/add?
graph_convolution_1/ReluRelugraph_convolution_1/add:z:0*
T0*4
_output_shapes"
 :??????????????????d2
graph_convolution_1/Relu?
,global_average_pooling1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,global_average_pooling1d/strided_slice/stack?
.global_average_pooling1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.global_average_pooling1d/strided_slice/stack_1?
.global_average_pooling1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.global_average_pooling1d/strided_slice/stack_2?
&global_average_pooling1d/strided_sliceStridedSlice&graph_convolution_1/Relu:activations:05global_average_pooling1d/strided_slice/stack:output:07global_average_pooling1d/strided_slice/stack_1:output:07global_average_pooling1d/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2(
&global_average_pooling1d/strided_slice?
global_average_pooling1d/CastCastinputs_1*

DstT0*

SrcT0
*0
_output_shapes
:??????????????????2
global_average_pooling1d/Cast?
'global_average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'global_average_pooling1d/ExpandDims/dim?
#global_average_pooling1d/ExpandDims
ExpandDims!global_average_pooling1d/Cast:y:00global_average_pooling1d/ExpandDims/dim:output:0*
T0*4
_output_shapes"
 :??????????????????2%
#global_average_pooling1d/ExpandDims?
global_average_pooling1d/mulMul&graph_convolution_1/Relu:activations:0,global_average_pooling1d/ExpandDims:output:0*
T0*4
_output_shapes"
 :??????????????????d2
global_average_pooling1d/mul?
.global_average_pooling1d/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :20
.global_average_pooling1d/Sum/reduction_indices?
global_average_pooling1d/SumSum global_average_pooling1d/mul:z:07global_average_pooling1d/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????d2
global_average_pooling1d/Sum?
0global_average_pooling1d/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :22
0global_average_pooling1d/Sum_1/reduction_indices?
global_average_pooling1d/Sum_1Sum,global_average_pooling1d/ExpandDims:output:09global_average_pooling1d/Sum_1/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2 
global_average_pooling1d/Sum_1?
 global_average_pooling1d/truedivRealDiv%global_average_pooling1d/Sum:output:0'global_average_pooling1d/Sum_1:output:0*
T0*'
_output_shapes
:?????????d2"
 global_average_pooling1d/truedivo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????d   2
flatten/Const?
flatten/ReshapeReshape$global_average_pooling1d/truediv:z:0flatten/Const:output:0*
T0*'
_output_shapes
:?????????d2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:d@*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_1/Relu?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_2/Relu?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAddy
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3/Sigmoidn
IdentityIdentitydense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp%^graph_convolution/add/ReadVariableOp+^graph_convolution/transpose/ReadVariableOp'^graph_convolution_1/add/ReadVariableOp-^graph_convolution_1/transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes
}:??????????????????o:??????????????????:'???????????????????????????: : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2L
$graph_convolution/add/ReadVariableOp$graph_convolution/add/ReadVariableOp2X
*graph_convolution/transpose/ReadVariableOp*graph_convolution/transpose/ReadVariableOp2P
&graph_convolution_1/add/ReadVariableOp&graph_convolution_1/add/ReadVariableOp2\
,graph_convolution_1/transpose/ReadVariableOp,graph_convolution_1/transpose/ReadVariableOp:^ Z
4
_output_shapes"
 :??????????????????o
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:??????????????????
"
_user_specified_name
inputs/1:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/2
?5
?
?__inference_model_layer_call_and_return_conditional_losses_5102
input_1
input_2

input_3(
graph_convolution_5068:od$
graph_convolution_5070:d*
graph_convolution_1_5074:dd&
graph_convolution_1_5076:d

dense_5081:d@

dense_5083:@
dense_1_5086:@ 
dense_1_5088: 
dense_2_5091: 
dense_2_5093:
dense_3_5096:
dense_3_5098:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?)graph_convolution/StatefulPartitionedCall?+graph_convolution_1/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????o* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_48862!
dropout/StatefulPartitionedCall?
)graph_convolution/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0input_3graph_convolution_5068graph_convolution_5070*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_graph_convolution_layer_call_and_return_conditional_losses_45962+
)graph_convolution/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall2graph_convolution/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_48522#
!dropout_1/StatefulPartitionedCall?
+graph_convolution_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0input_3graph_convolution_1_5074graph_convolution_1_5076*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_graph_convolution_1_layer_call_and_return_conditional_losses_46402-
+graph_convolution_1/StatefulPartitionedCall?
(global_average_pooling1d/PartitionedCallPartitionedCall4graph_convolution_1/StatefulPartitionedCall:output:0input_2*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_46632*
(global_average_pooling1d/PartitionedCall?
flatten/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_46712
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dense_5081
dense_5083*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_46842
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_5086dense_1_5088*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_47012!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_5091dense_2_5093*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_47182!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_5096dense_3_5098*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_47352!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*^graph_convolution/StatefulPartitionedCall,^graph_convolution_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes
}:??????????????????o:??????????????????:'???????????????????????????: : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2V
)graph_convolution/StatefulPartitionedCall)graph_convolution/StatefulPartitionedCall2Z
+graph_convolution_1/StatefulPartitionedCall+graph_convolution_1/StatefulPartitionedCall:] Y
4
_output_shapes"
 :??????????????????o
!
_user_specified_name	input_1:YU
0
_output_shapes
:??????????????????
!
_user_specified_name	input_2:fb
=
_output_shapes+
):'???????????????????????????
!
_user_specified_name	input_3
?
?
K__inference_graph_convolution_layer_call_and_return_conditional_losses_5481
inputs_0
inputs_11
shape_2_readvariableop_resource:od)
add_readvariableop_resource:d
identity??add/ReadVariableOp?transpose/ReadVariableOpt
MatMulBatchMatMulV2inputs_1inputs_0*
T0*4
_output_shapes"
 :??????????????????o2
MatMulM
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:2
ShapeQ
Shape_1ShapeMatMul:output:0*
T0*
_output_shapes
:2	
Shape_1^
unstackUnpackShape_1:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack?
Shape_2/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

:od*
dtype02
Shape_2/ReadVariableOpc
Shape_2Const*
_output_shapes
:*
dtype0*
valueB"o   d   2	
Shape_2`
	unstack_1UnpackShape_2:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????o   2
Reshape/shapex
ReshapeReshapeMatMul:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????o2	
Reshape?
transpose/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

:od*
dtype02
transpose/ReadVariableOpq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm?
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:od2
	transposes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"o   ????2
Reshape_1/shapes
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:od2
	Reshape_1v
MatMul_1MatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:?????????d2

MatMul_1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d2
Reshape_2/shape/2?
Reshape_2/shapePackunstack:output:0unstack:output:1Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape?
	Reshape_2ReshapeMatMul_1:product:0Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :??????????????????d2
	Reshape_2?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:d*
dtype02
add/ReadVariableOp?
addAddV2Reshape_2:output:0add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????d2
add\
ReluReluadd:z:0*
T0*4
_output_shapes"
 :??????????????????d2
Reluz
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :??????????????????d2

Identity~
NoOpNoOp^add/ReadVariableOp^transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:??????????????????o:'???????????????????????????: : 2(
add/ReadVariableOpadd/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp:^ Z
4
_output_shapes"
 :??????????????????o
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/1
?
?
K__inference_graph_convolution_layer_call_and_return_conditional_losses_4596

inputs
inputs_11
shape_2_readvariableop_resource:od)
add_readvariableop_resource:d
identity??add/ReadVariableOp?transpose/ReadVariableOpr
MatMulBatchMatMulV2inputs_1inputs*
T0*4
_output_shapes"
 :??????????????????o2
MatMulM
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:2
ShapeQ
Shape_1ShapeMatMul:output:0*
T0*
_output_shapes
:2	
Shape_1^
unstackUnpackShape_1:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack?
Shape_2/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

:od*
dtype02
Shape_2/ReadVariableOpc
Shape_2Const*
_output_shapes
:*
dtype0*
valueB"o   d   2	
Shape_2`
	unstack_1UnpackShape_2:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????o   2
Reshape/shapex
ReshapeReshapeMatMul:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????o2	
Reshape?
transpose/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

:od*
dtype02
transpose/ReadVariableOpq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm?
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:od2
	transposes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"o   ????2
Reshape_1/shapes
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:od2
	Reshape_1v
MatMul_1MatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:?????????d2

MatMul_1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d2
Reshape_2/shape/2?
Reshape_2/shapePackunstack:output:0unstack:output:1Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape?
	Reshape_2ReshapeMatMul_1:product:0Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :??????????????????d2
	Reshape_2?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:d*
dtype02
add/ReadVariableOp?
addAddV2Reshape_2:output:0add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????d2
add\
ReluReluadd:z:0*
T0*4
_output_shapes"
 :??????????????????d2
Reluz
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :??????????????????d2

Identity~
NoOpNoOp^add/ReadVariableOp^transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:??????????????????o:'???????????????????????????: : 2(
add/ReadVariableOpadd/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????o
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_4607

inputs

identity_1g
IdentityIdentityinputs*
T0*4
_output_shapes"
 :??????????????????d2

Identityv

Identity_1IdentityIdentity:output:0*
T0*4
_output_shapes"
 :??????????????????d2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????????????d:\ X
4
_output_shapes"
 :??????????????????d
 
_user_specified_nameinputs
?
_
A__inference_dropout_layer_call_and_return_conditional_losses_5428

inputs

identity_1g
IdentityIdentityinputs*
T0*4
_output_shapes"
 :??????????????????o2

Identityv

Identity_1IdentityIdentity:output:0*
T0*4
_output_shapes"
 :??????????????????o2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????????????o:\ X
4
_output_shapes"
 :??????????????????o
 
_user_specified_nameinputs
?
D
(__inference_dropout_1_layer_call_fn_5513

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_46072
PartitionedCally
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :??????????????????d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????????????d:\ X
4
_output_shapes"
 :??????????????????d
 
_user_specified_nameinputs
?
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_5496

inputs

identity_1g
IdentityIdentityinputs*
T0*4
_output_shapes"
 :??????????????????d2

Identityv

Identity_1IdentityIdentity:output:0*
T0*4
_output_shapes"
 :??????????????????d2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????????????d:\ X
4
_output_shapes"
 :??????????????????d
 
_user_specified_nameinputs
?
_
A__inference_dropout_layer_call_and_return_conditional_losses_4563

inputs

identity_1g
IdentityIdentityinputs*
T0*4
_output_shapes"
 :??????????????????o2

Identityv

Identity_1IdentityIdentity:output:0*
T0*4
_output_shapes"
 :??????????????????o2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????????????o:\ X
4
_output_shapes"
 :??????????????????o
 
_user_specified_nameinputs
??
?
 __inference__traced_restore_5990
file_prefix;
)assignvariableop_graph_convolution_kernel:od7
)assignvariableop_1_graph_convolution_bias:d?
-assignvariableop_2_graph_convolution_1_kernel:dd9
+assignvariableop_3_graph_convolution_1_bias:d1
assignvariableop_4_dense_kernel:d@+
assignvariableop_5_dense_bias:@3
!assignvariableop_6_dense_1_kernel:@ -
assignvariableop_7_dense_1_bias: 3
!assignvariableop_8_dense_2_kernel: -
assignvariableop_9_dense_2_bias:4
"assignvariableop_10_dense_3_kernel:.
 assignvariableop_11_dense_3_bias:'
assignvariableop_12_adam_iter:	 )
assignvariableop_13_adam_beta_1: )
assignvariableop_14_adam_beta_2: (
assignvariableop_15_adam_decay: 0
&assignvariableop_16_adam_learning_rate: #
assignvariableop_17_total: #
assignvariableop_18_count: %
assignvariableop_19_total_1: %
assignvariableop_20_count_1: E
3assignvariableop_21_adam_graph_convolution_kernel_m:od?
1assignvariableop_22_adam_graph_convolution_bias_m:dG
5assignvariableop_23_adam_graph_convolution_1_kernel_m:ddA
3assignvariableop_24_adam_graph_convolution_1_bias_m:d9
'assignvariableop_25_adam_dense_kernel_m:d@3
%assignvariableop_26_adam_dense_bias_m:@;
)assignvariableop_27_adam_dense_1_kernel_m:@ 5
'assignvariableop_28_adam_dense_1_bias_m: ;
)assignvariableop_29_adam_dense_2_kernel_m: 5
'assignvariableop_30_adam_dense_2_bias_m:;
)assignvariableop_31_adam_dense_3_kernel_m:5
'assignvariableop_32_adam_dense_3_bias_m:E
3assignvariableop_33_adam_graph_convolution_kernel_v:od?
1assignvariableop_34_adam_graph_convolution_bias_v:dG
5assignvariableop_35_adam_graph_convolution_1_kernel_v:ddA
3assignvariableop_36_adam_graph_convolution_1_bias_v:d9
'assignvariableop_37_adam_dense_kernel_v:d@3
%assignvariableop_38_adam_dense_bias_v:@;
)assignvariableop_39_adam_dense_1_kernel_v:@ 5
'assignvariableop_40_adam_dense_1_bias_v: ;
)assignvariableop_41_adam_dense_2_kernel_v: 5
'assignvariableop_42_adam_dense_2_bias_v:;
)assignvariableop_43_adam_dense_3_kernel_v:5
'assignvariableop_44_adam_dense_3_bias_v:
identity_46??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp)assignvariableop_graph_convolution_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp)assignvariableop_1_graph_convolution_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp-assignvariableop_2_graph_convolution_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp+assignvariableop_3_graph_convolution_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_3_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_3_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp3assignvariableop_21_adam_graph_convolution_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp1assignvariableop_22_adam_graph_convolution_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp5assignvariableop_23_adam_graph_convolution_1_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp3assignvariableop_24_adam_graph_convolution_1_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp'assignvariableop_25_adam_dense_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp%assignvariableop_26_adam_dense_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_dense_1_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adam_dense_1_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_2_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_2_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_3_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_3_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp3assignvariableop_33_adam_graph_convolution_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp1assignvariableop_34_adam_graph_convolution_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp5assignvariableop_35_adam_graph_convolution_1_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp3assignvariableop_36_adam_graph_convolution_1_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp'assignvariableop_37_adam_dense_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp%assignvariableop_38_adam_dense_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_dense_1_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp'assignvariableop_40_adam_dense_1_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_dense_2_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adam_dense_2_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_dense_3_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_dense_3_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_449
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_45f
Identity_46IdentityIdentity_45:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_46?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_46Identity_46:output:0*o
_input_shapes^
\: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
"__inference_signature_wrapper_5141
input_1
input_2

input_3
unknown:od
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:d@
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7: 
	unknown_8:
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_45232
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes
}:??????????????????o:??????????????????:'???????????????????????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :??????????????????o
!
_user_specified_name	input_1:YU
0
_output_shapes
:??????????????????
!
_user_specified_name	input_2:fb
=
_output_shapes+
):'???????????????????????????
!
_user_specified_name	input_3
?2
?
?__inference_model_layer_call_and_return_conditional_losses_5062
input_1
input_2

input_3(
graph_convolution_5028:od$
graph_convolution_5030:d*
graph_convolution_1_5034:dd&
graph_convolution_1_5036:d

dense_5041:d@

dense_5043:@
dense_1_5046:@ 
dense_1_5048: 
dense_2_5051: 
dense_2_5053:
dense_3_5056:
dense_3_5058:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?)graph_convolution/StatefulPartitionedCall?+graph_convolution_1/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????o* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_45632
dropout/PartitionedCall?
)graph_convolution/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0input_3graph_convolution_5028graph_convolution_5030*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_graph_convolution_layer_call_and_return_conditional_losses_45962+
)graph_convolution/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall2graph_convolution/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_46072
dropout_1/PartitionedCall?
+graph_convolution_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0input_3graph_convolution_1_5034graph_convolution_1_5036*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_graph_convolution_1_layer_call_and_return_conditional_losses_46402-
+graph_convolution_1/StatefulPartitionedCall?
(global_average_pooling1d/PartitionedCallPartitionedCall4graph_convolution_1/StatefulPartitionedCall:output:0input_2*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_46632*
(global_average_pooling1d/PartitionedCall?
flatten/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_46712
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dense_5041
dense_5043*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_46842
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_5046dense_1_5048*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_47012!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_5051dense_2_5053*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_47182!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_5056dense_3_5058*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_47352!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*^graph_convolution/StatefulPartitionedCall,^graph_convolution_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes
}:??????????????????o:??????????????????:'???????????????????????????: : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2V
)graph_convolution/StatefulPartitionedCall)graph_convolution/StatefulPartitionedCall2Z
+graph_convolution_1/StatefulPartitionedCall+graph_convolution_1/StatefulPartitionedCall:] Y
4
_output_shapes"
 :??????????????????o
!
_user_specified_name	input_1:YU
0
_output_shapes
:??????????????????
!
_user_specified_name	input_2:fb
=
_output_shapes+
):'???????????????????????????
!
_user_specified_name	input_3
?

?
A__inference_dense_2_layer_call_and_return_conditional_losses_5656

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
n
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_4533

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
&__inference_dense_2_layer_call_fn_5665

inputs
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_47182
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_5022
input_1
input_2

input_3
unknown:od
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:d@
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7: 
	unknown_8:
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_49642
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes
}:??????????????????o:??????????????????:'???????????????????????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :??????????????????o
!
_user_specified_name	input_1:YU
0
_output_shapes
:??????????????????
!
_user_specified_name	input_2:fb
=
_output_shapes+
):'???????????????????????????
!
_user_specified_name	input_3
?
?
A__inference_dense_3_layer_call_and_return_conditional_losses_5676

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
B
&__inference_flatten_layer_call_fn_5605

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_46712
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?

?
A__inference_dense_1_layer_call_and_return_conditional_losses_5636

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?5
?
?__inference_model_layer_call_and_return_conditional_losses_4964

inputs
inputs_1

inputs_2(
graph_convolution_4930:od$
graph_convolution_4932:d*
graph_convolution_1_4936:dd&
graph_convolution_1_4938:d

dense_4943:d@

dense_4945:@
dense_1_4948:@ 
dense_1_4950: 
dense_2_4953: 
dense_2_4955:
dense_3_4958:
dense_3_4960:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?)graph_convolution/StatefulPartitionedCall?+graph_convolution_1/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????o* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_48862!
dropout/StatefulPartitionedCall?
)graph_convolution/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0inputs_2graph_convolution_4930graph_convolution_4932*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_graph_convolution_layer_call_and_return_conditional_losses_45962+
)graph_convolution/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall2graph_convolution/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_48522#
!dropout_1/StatefulPartitionedCall?
+graph_convolution_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0inputs_2graph_convolution_1_4936graph_convolution_1_4938*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_graph_convolution_1_layer_call_and_return_conditional_losses_46402-
+graph_convolution_1/StatefulPartitionedCall?
(global_average_pooling1d/PartitionedCallPartitionedCall4graph_convolution_1/StatefulPartitionedCall:output:0inputs_1*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_46632*
(global_average_pooling1d/PartitionedCall?
flatten/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_46712
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dense_4943
dense_4945*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_46842
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_4948dense_1_4950*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_47012!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_4953dense_2_4955*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_47182!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_4958dense_3_4960*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_47352!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*^graph_convolution/StatefulPartitionedCall,^graph_convolution_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes
}:??????????????????o:??????????????????:'???????????????????????????: : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2V
)graph_convolution/StatefulPartitionedCall)graph_convolution/StatefulPartitionedCall2Z
+graph_convolution_1/StatefulPartitionedCall+graph_convolution_1/StatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????o
 
_user_specified_nameinputs:XT
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?

?
?__inference_dense_layer_call_and_return_conditional_losses_4684

inputs0
matmul_readvariableop_resource:d@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?

?
A__inference_dense_1_layer_call_and_return_conditional_losses_4701

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
`
A__inference_dropout_layer_call_and_return_conditional_losses_4886

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const?
dropout/MulMulinputsdropout/Const:output:0*
T0*4
_output_shapes"
 :??????????????????o2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*4
_output_shapes"
 :??????????????????o*
dtype0*
seed?'2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????o2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????o2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*4
_output_shapes"
 :??????????????????o2
dropout/Mul_1r
IdentityIdentitydropout/Mul_1:z:0*
T0*4
_output_shapes"
 :??????????????????o2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????????????o:\ X
4
_output_shapes"
 :??????????????????o
 
_user_specified_nameinputs
?
n
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_5565

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
]
A__inference_flatten_layer_call_and_return_conditional_losses_5600

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????d   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????d2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
&__inference_dense_1_layer_call_fn_5645

inputs
unknown:@ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_47012
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?

__inference__wrapped_model_4523
input_1
input_2

input_3I
7model_graph_convolution_shape_2_readvariableop_resource:odA
3model_graph_convolution_add_readvariableop_resource:dK
9model_graph_convolution_1_shape_2_readvariableop_resource:ddC
5model_graph_convolution_1_add_readvariableop_resource:d<
*model_dense_matmul_readvariableop_resource:d@9
+model_dense_biasadd_readvariableop_resource:@>
,model_dense_1_matmul_readvariableop_resource:@ ;
-model_dense_1_biasadd_readvariableop_resource: >
,model_dense_2_matmul_readvariableop_resource: ;
-model_dense_2_biasadd_readvariableop_resource:>
,model_dense_3_matmul_readvariableop_resource:;
-model_dense_3_biasadd_readvariableop_resource:
identity??"model/dense/BiasAdd/ReadVariableOp?!model/dense/MatMul/ReadVariableOp?$model/dense_1/BiasAdd/ReadVariableOp?#model/dense_1/MatMul/ReadVariableOp?$model/dense_2/BiasAdd/ReadVariableOp?#model/dense_2/MatMul/ReadVariableOp?$model/dense_3/BiasAdd/ReadVariableOp?#model/dense_3/MatMul/ReadVariableOp?*model/graph_convolution/add/ReadVariableOp?0model/graph_convolution/transpose/ReadVariableOp?,model/graph_convolution_1/add/ReadVariableOp?2model/graph_convolution_1/transpose/ReadVariableOp?
model/dropout/IdentityIdentityinput_1*
T0*4
_output_shapes"
 :??????????????????o2
model/dropout/Identity?
model/graph_convolution/MatMulBatchMatMulV2input_3model/dropout/Identity:output:0*
T0*4
_output_shapes"
 :??????????????????o2 
model/graph_convolution/MatMul?
model/graph_convolution/ShapeShape'model/graph_convolution/MatMul:output:0*
T0*
_output_shapes
:2
model/graph_convolution/Shape?
model/graph_convolution/Shape_1Shape'model/graph_convolution/MatMul:output:0*
T0*
_output_shapes
:2!
model/graph_convolution/Shape_1?
model/graph_convolution/unstackUnpack(model/graph_convolution/Shape_1:output:0*
T0*
_output_shapes
: : : *	
num2!
model/graph_convolution/unstack?
.model/graph_convolution/Shape_2/ReadVariableOpReadVariableOp7model_graph_convolution_shape_2_readvariableop_resource*
_output_shapes

:od*
dtype020
.model/graph_convolution/Shape_2/ReadVariableOp?
model/graph_convolution/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"o   d   2!
model/graph_convolution/Shape_2?
!model/graph_convolution/unstack_1Unpack(model/graph_convolution/Shape_2:output:0*
T0*
_output_shapes
: : *	
num2#
!model/graph_convolution/unstack_1?
%model/graph_convolution/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????o   2'
%model/graph_convolution/Reshape/shape?
model/graph_convolution/ReshapeReshape'model/graph_convolution/MatMul:output:0.model/graph_convolution/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????o2!
model/graph_convolution/Reshape?
0model/graph_convolution/transpose/ReadVariableOpReadVariableOp7model_graph_convolution_shape_2_readvariableop_resource*
_output_shapes

:od*
dtype022
0model/graph_convolution/transpose/ReadVariableOp?
&model/graph_convolution/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2(
&model/graph_convolution/transpose/perm?
!model/graph_convolution/transpose	Transpose8model/graph_convolution/transpose/ReadVariableOp:value:0/model/graph_convolution/transpose/perm:output:0*
T0*
_output_shapes

:od2#
!model/graph_convolution/transpose?
'model/graph_convolution/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"o   ????2)
'model/graph_convolution/Reshape_1/shape?
!model/graph_convolution/Reshape_1Reshape%model/graph_convolution/transpose:y:00model/graph_convolution/Reshape_1/shape:output:0*
T0*
_output_shapes

:od2#
!model/graph_convolution/Reshape_1?
 model/graph_convolution/MatMul_1MatMul(model/graph_convolution/Reshape:output:0*model/graph_convolution/Reshape_1:output:0*
T0*'
_output_shapes
:?????????d2"
 model/graph_convolution/MatMul_1?
)model/graph_convolution/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d2+
)model/graph_convolution/Reshape_2/shape/2?
'model/graph_convolution/Reshape_2/shapePack(model/graph_convolution/unstack:output:0(model/graph_convolution/unstack:output:12model/graph_convolution/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2)
'model/graph_convolution/Reshape_2/shape?
!model/graph_convolution/Reshape_2Reshape*model/graph_convolution/MatMul_1:product:00model/graph_convolution/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :??????????????????d2#
!model/graph_convolution/Reshape_2?
*model/graph_convolution/add/ReadVariableOpReadVariableOp3model_graph_convolution_add_readvariableop_resource*
_output_shapes
:d*
dtype02,
*model/graph_convolution/add/ReadVariableOp?
model/graph_convolution/addAddV2*model/graph_convolution/Reshape_2:output:02model/graph_convolution/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????d2
model/graph_convolution/add?
model/graph_convolution/ReluRelumodel/graph_convolution/add:z:0*
T0*4
_output_shapes"
 :??????????????????d2
model/graph_convolution/Relu?
model/dropout_1/IdentityIdentity*model/graph_convolution/Relu:activations:0*
T0*4
_output_shapes"
 :??????????????????d2
model/dropout_1/Identity?
 model/graph_convolution_1/MatMulBatchMatMulV2input_3!model/dropout_1/Identity:output:0*
T0*4
_output_shapes"
 :??????????????????d2"
 model/graph_convolution_1/MatMul?
model/graph_convolution_1/ShapeShape)model/graph_convolution_1/MatMul:output:0*
T0*
_output_shapes
:2!
model/graph_convolution_1/Shape?
!model/graph_convolution_1/Shape_1Shape)model/graph_convolution_1/MatMul:output:0*
T0*
_output_shapes
:2#
!model/graph_convolution_1/Shape_1?
!model/graph_convolution_1/unstackUnpack*model/graph_convolution_1/Shape_1:output:0*
T0*
_output_shapes
: : : *	
num2#
!model/graph_convolution_1/unstack?
0model/graph_convolution_1/Shape_2/ReadVariableOpReadVariableOp9model_graph_convolution_1_shape_2_readvariableop_resource*
_output_shapes

:dd*
dtype022
0model/graph_convolution_1/Shape_2/ReadVariableOp?
!model/graph_convolution_1/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"d   d   2#
!model/graph_convolution_1/Shape_2?
#model/graph_convolution_1/unstack_1Unpack*model/graph_convolution_1/Shape_2:output:0*
T0*
_output_shapes
: : *	
num2%
#model/graph_convolution_1/unstack_1?
'model/graph_convolution_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2)
'model/graph_convolution_1/Reshape/shape?
!model/graph_convolution_1/ReshapeReshape)model/graph_convolution_1/MatMul:output:00model/graph_convolution_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????d2#
!model/graph_convolution_1/Reshape?
2model/graph_convolution_1/transpose/ReadVariableOpReadVariableOp9model_graph_convolution_1_shape_2_readvariableop_resource*
_output_shapes

:dd*
dtype024
2model/graph_convolution_1/transpose/ReadVariableOp?
(model/graph_convolution_1/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2*
(model/graph_convolution_1/transpose/perm?
#model/graph_convolution_1/transpose	Transpose:model/graph_convolution_1/transpose/ReadVariableOp:value:01model/graph_convolution_1/transpose/perm:output:0*
T0*
_output_shapes

:dd2%
#model/graph_convolution_1/transpose?
)model/graph_convolution_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"d   ????2+
)model/graph_convolution_1/Reshape_1/shape?
#model/graph_convolution_1/Reshape_1Reshape'model/graph_convolution_1/transpose:y:02model/graph_convolution_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:dd2%
#model/graph_convolution_1/Reshape_1?
"model/graph_convolution_1/MatMul_1MatMul*model/graph_convolution_1/Reshape:output:0,model/graph_convolution_1/Reshape_1:output:0*
T0*'
_output_shapes
:?????????d2$
"model/graph_convolution_1/MatMul_1?
+model/graph_convolution_1/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d2-
+model/graph_convolution_1/Reshape_2/shape/2?
)model/graph_convolution_1/Reshape_2/shapePack*model/graph_convolution_1/unstack:output:0*model/graph_convolution_1/unstack:output:14model/graph_convolution_1/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2+
)model/graph_convolution_1/Reshape_2/shape?
#model/graph_convolution_1/Reshape_2Reshape,model/graph_convolution_1/MatMul_1:product:02model/graph_convolution_1/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :??????????????????d2%
#model/graph_convolution_1/Reshape_2?
,model/graph_convolution_1/add/ReadVariableOpReadVariableOp5model_graph_convolution_1_add_readvariableop_resource*
_output_shapes
:d*
dtype02.
,model/graph_convolution_1/add/ReadVariableOp?
model/graph_convolution_1/addAddV2,model/graph_convolution_1/Reshape_2:output:04model/graph_convolution_1/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????d2
model/graph_convolution_1/add?
model/graph_convolution_1/ReluRelu!model/graph_convolution_1/add:z:0*
T0*4
_output_shapes"
 :??????????????????d2 
model/graph_convolution_1/Relu?
2model/global_average_pooling1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2model/global_average_pooling1d/strided_slice/stack?
4model/global_average_pooling1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4model/global_average_pooling1d/strided_slice/stack_1?
4model/global_average_pooling1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4model/global_average_pooling1d/strided_slice/stack_2?
,model/global_average_pooling1d/strided_sliceStridedSlice,model/graph_convolution_1/Relu:activations:0;model/global_average_pooling1d/strided_slice/stack:output:0=model/global_average_pooling1d/strided_slice/stack_1:output:0=model/global_average_pooling1d/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2.
,model/global_average_pooling1d/strided_slice?
#model/global_average_pooling1d/CastCastinput_2*

DstT0*

SrcT0
*0
_output_shapes
:??????????????????2%
#model/global_average_pooling1d/Cast?
-model/global_average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-model/global_average_pooling1d/ExpandDims/dim?
)model/global_average_pooling1d/ExpandDims
ExpandDims'model/global_average_pooling1d/Cast:y:06model/global_average_pooling1d/ExpandDims/dim:output:0*
T0*4
_output_shapes"
 :??????????????????2+
)model/global_average_pooling1d/ExpandDims?
"model/global_average_pooling1d/mulMul,model/graph_convolution_1/Relu:activations:02model/global_average_pooling1d/ExpandDims:output:0*
T0*4
_output_shapes"
 :??????????????????d2$
"model/global_average_pooling1d/mul?
4model/global_average_pooling1d/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :26
4model/global_average_pooling1d/Sum/reduction_indices?
"model/global_average_pooling1d/SumSum&model/global_average_pooling1d/mul:z:0=model/global_average_pooling1d/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????d2$
"model/global_average_pooling1d/Sum?
6model/global_average_pooling1d/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :28
6model/global_average_pooling1d/Sum_1/reduction_indices?
$model/global_average_pooling1d/Sum_1Sum2model/global_average_pooling1d/ExpandDims:output:0?model/global_average_pooling1d/Sum_1/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2&
$model/global_average_pooling1d/Sum_1?
&model/global_average_pooling1d/truedivRealDiv+model/global_average_pooling1d/Sum:output:0-model/global_average_pooling1d/Sum_1:output:0*
T0*'
_output_shapes
:?????????d2(
&model/global_average_pooling1d/truediv{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????d   2
model/flatten/Const?
model/flatten/ReshapeReshape*model/global_average_pooling1d/truediv:z:0model/flatten/Const:output:0*
T0*'
_output_shapes
:?????????d2
model/flatten/Reshape?
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

:d@*
dtype02#
!model/dense/MatMul/ReadVariableOp?
model/dense/MatMulMatMulmodel/flatten/Reshape:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model/dense/MatMul?
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02$
"model/dense/BiasAdd/ReadVariableOp?
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model/dense/BiasAdd|
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model/dense/Relu?
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02%
#model/dense_1/MatMul/ReadVariableOp?
model/dense_1/MatMulMatMulmodel/dense/Relu:activations:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
model/dense_1/MatMul?
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$model/dense_1/BiasAdd/ReadVariableOp?
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
model/dense_1/BiasAdd?
model/dense_1/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
model/dense_1/Relu?
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02%
#model/dense_2/MatMul/ReadVariableOp?
model/dense_2/MatMulMatMul model/dense_1/Relu:activations:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense_2/MatMul?
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense_2/BiasAdd/ReadVariableOp?
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense_2/BiasAdd?
model/dense_2/ReluRelumodel/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model/dense_2/Relu?
#model/dense_3/MatMul/ReadVariableOpReadVariableOp,model_dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#model/dense_3/MatMul/ReadVariableOp?
model/dense_3/MatMulMatMul model/dense_2/Relu:activations:0+model/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense_3/MatMul?
$model/dense_3/BiasAdd/ReadVariableOpReadVariableOp-model_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense_3/BiasAdd/ReadVariableOp?
model/dense_3/BiasAddBiasAddmodel/dense_3/MatMul:product:0,model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense_3/BiasAdd?
model/dense_3/SigmoidSigmoidmodel/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model/dense_3/Sigmoidt
IdentityIdentitymodel/dense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp%^model/dense_3/BiasAdd/ReadVariableOp$^model/dense_3/MatMul/ReadVariableOp+^model/graph_convolution/add/ReadVariableOp1^model/graph_convolution/transpose/ReadVariableOp-^model/graph_convolution_1/add/ReadVariableOp3^model/graph_convolution_1/transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes
}:??????????????????o:??????????????????:'???????????????????????????: : : : : : : : : : : : 2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp2L
$model/dense_3/BiasAdd/ReadVariableOp$model/dense_3/BiasAdd/ReadVariableOp2J
#model/dense_3/MatMul/ReadVariableOp#model/dense_3/MatMul/ReadVariableOp2X
*model/graph_convolution/add/ReadVariableOp*model/graph_convolution/add/ReadVariableOp2d
0model/graph_convolution/transpose/ReadVariableOp0model/graph_convolution/transpose/ReadVariableOp2\
,model/graph_convolution_1/add/ReadVariableOp,model/graph_convolution_1/add/ReadVariableOp2h
2model/graph_convolution_1/transpose/ReadVariableOp2model/graph_convolution_1/transpose/ReadVariableOp:] Y
4
_output_shapes"
 :??????????????????o
!
_user_specified_name	input_1:YU
0
_output_shapes
:??????????????????
!
_user_specified_name	input_2:fb
=
_output_shapes+
):'???????????????????????????
!
_user_specified_name	input_3
?

?
?__inference_dense_layer_call_and_return_conditional_losses_5616

inputs0
matmul_readvariableop_resource:d@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?	
?
0__inference_graph_convolution_layer_call_fn_5491
inputs_0
inputs_1
unknown:od
	unknown_0:d
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_graph_convolution_layer_call_and_return_conditional_losses_45962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????d2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:??????????????????o:'???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????o
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/1
?
a
(__inference_dropout_1_layer_call_fn_5518

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_48522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????d2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????????????d22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????d
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_5423
inputs_0
inputs_1

inputs_2
unknown:od
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:d@
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7: 
	unknown_8:
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_49642
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes
}:??????????????????o:??????????????????:'???????????????????????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????o
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:??????????????????
"
_user_specified_name
inputs/1:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/2
?
_
&__inference_dropout_layer_call_fn_5450

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????o* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_48862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????o2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????????????o22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????o
 
_user_specified_nameinputs
?
B
&__inference_dropout_layer_call_fn_5445

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????o* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_45632
PartitionedCally
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :??????????????????o2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????????????o:\ X
4
_output_shapes"
 :??????????????????o
 
_user_specified_nameinputs
?^
?
__inference__traced_save_5845
file_prefix7
3savev2_graph_convolution_kernel_read_readvariableop5
1savev2_graph_convolution_bias_read_readvariableop9
5savev2_graph_convolution_1_kernel_read_readvariableop7
3savev2_graph_convolution_1_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop>
:savev2_adam_graph_convolution_kernel_m_read_readvariableop<
8savev2_adam_graph_convolution_bias_m_read_readvariableop@
<savev2_adam_graph_convolution_1_kernel_m_read_readvariableop>
:savev2_adam_graph_convolution_1_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableop>
:savev2_adam_graph_convolution_kernel_v_read_readvariableop<
8savev2_adam_graph_convolution_bias_v_read_readvariableop@
<savev2_adam_graph_convolution_1_kernel_v_read_readvariableop>
:savev2_adam_graph_convolution_1_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:03savev2_graph_convolution_kernel_read_readvariableop1savev2_graph_convolution_bias_read_readvariableop5savev2_graph_convolution_1_kernel_read_readvariableop3savev2_graph_convolution_1_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop:savev2_adam_graph_convolution_kernel_m_read_readvariableop8savev2_adam_graph_convolution_bias_m_read_readvariableop<savev2_adam_graph_convolution_1_kernel_m_read_readvariableop:savev2_adam_graph_convolution_1_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop:savev2_adam_graph_convolution_kernel_v_read_readvariableop8savev2_adam_graph_convolution_bias_v_read_readvariableop<savev2_adam_graph_convolution_1_kernel_v_read_readvariableop:savev2_adam_graph_convolution_1_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :od:d:dd:d:d@:@:@ : : :::: : : : : : : : : :od:d:dd:d:d@:@:@ : : ::::od:d:dd:d:d@:@:@ : : :::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:od: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:d@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$	 

_output_shapes

: : 


_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:od: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:d@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$  

_output_shapes

:: !

_output_shapes
::$" 

_output_shapes

:od: #

_output_shapes
:d:$$ 

_output_shapes

:dd: %

_output_shapes
:d:$& 

_output_shapes

:d@: '

_output_shapes
:@:$( 

_output_shapes

:@ : )

_output_shapes
: :$* 

_output_shapes

: : +

_output_shapes
::$, 

_output_shapes

:: -

_output_shapes
::.

_output_shapes
: 
?
?
&__inference_dense_3_layer_call_fn_5685

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_47352
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
x
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_5583

inputs
mask

identityt
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2
strided_sliced
CastCastmask*

DstT0*

SrcT0
*0
_output_shapes
:??????????????????2
Castb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsCast:y:0ExpandDims/dim:output:0*
T0*4
_output_shapes"
 :??????????????????2

ExpandDimsm
mulMulinputsExpandDims:output:0*
T0*4
_output_shapes"
 :??????????????????d2
mulp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesl
SumSummul:z:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????d2
Sumt
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_1/reduction_indices~
Sum_1SumExpandDims:output:0 Sum_1/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
Sum_1m
truedivRealDivSum:output:0Sum_1:output:0*
T0*'
_output_shapes
:?????????d2	
truediv_
IdentityIdentitytruediv:z:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:??????????????????d:??????????????????:\ X
4
_output_shapes"
 :??????????????????d
 
_user_specified_nameinputs:VR
0
_output_shapes
:??????????????????

_user_specified_namemask
??
?	
?__inference_model_layer_call_and_return_conditional_losses_5361
inputs_0
inputs_1

inputs_2C
1graph_convolution_shape_2_readvariableop_resource:od;
-graph_convolution_add_readvariableop_resource:dE
3graph_convolution_1_shape_2_readvariableop_resource:dd=
/graph_convolution_1_add_readvariableop_resource:d6
$dense_matmul_readvariableop_resource:d@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@ 5
'dense_1_biasadd_readvariableop_resource: 8
&dense_2_matmul_readvariableop_resource: 5
'dense_2_biasadd_readvariableop_resource:8
&dense_3_matmul_readvariableop_resource:5
'dense_3_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?$graph_convolution/add/ReadVariableOp?*graph_convolution/transpose/ReadVariableOp?&graph_convolution_1/add/ReadVariableOp?,graph_convolution_1/transpose/ReadVariableOps
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/Const?
dropout/dropout/MulMulinputs_0dropout/dropout/Const:output:0*
T0*4
_output_shapes"
 :??????????????????o2
dropout/dropout/Mulf
dropout/dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*4
_output_shapes"
 :??????????????????o*
dtype0*
seed?'2.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????o2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????o2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*4
_output_shapes"
 :??????????????????o2
dropout/dropout/Mul_1?
graph_convolution/MatMulBatchMatMulV2inputs_2dropout/dropout/Mul_1:z:0*
T0*4
_output_shapes"
 :??????????????????o2
graph_convolution/MatMul?
graph_convolution/ShapeShape!graph_convolution/MatMul:output:0*
T0*
_output_shapes
:2
graph_convolution/Shape?
graph_convolution/Shape_1Shape!graph_convolution/MatMul:output:0*
T0*
_output_shapes
:2
graph_convolution/Shape_1?
graph_convolution/unstackUnpack"graph_convolution/Shape_1:output:0*
T0*
_output_shapes
: : : *	
num2
graph_convolution/unstack?
(graph_convolution/Shape_2/ReadVariableOpReadVariableOp1graph_convolution_shape_2_readvariableop_resource*
_output_shapes

:od*
dtype02*
(graph_convolution/Shape_2/ReadVariableOp?
graph_convolution/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"o   d   2
graph_convolution/Shape_2?
graph_convolution/unstack_1Unpack"graph_convolution/Shape_2:output:0*
T0*
_output_shapes
: : *	
num2
graph_convolution/unstack_1?
graph_convolution/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????o   2!
graph_convolution/Reshape/shape?
graph_convolution/ReshapeReshape!graph_convolution/MatMul:output:0(graph_convolution/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????o2
graph_convolution/Reshape?
*graph_convolution/transpose/ReadVariableOpReadVariableOp1graph_convolution_shape_2_readvariableop_resource*
_output_shapes

:od*
dtype02,
*graph_convolution/transpose/ReadVariableOp?
 graph_convolution/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2"
 graph_convolution/transpose/perm?
graph_convolution/transpose	Transpose2graph_convolution/transpose/ReadVariableOp:value:0)graph_convolution/transpose/perm:output:0*
T0*
_output_shapes

:od2
graph_convolution/transpose?
!graph_convolution/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"o   ????2#
!graph_convolution/Reshape_1/shape?
graph_convolution/Reshape_1Reshapegraph_convolution/transpose:y:0*graph_convolution/Reshape_1/shape:output:0*
T0*
_output_shapes

:od2
graph_convolution/Reshape_1?
graph_convolution/MatMul_1MatMul"graph_convolution/Reshape:output:0$graph_convolution/Reshape_1:output:0*
T0*'
_output_shapes
:?????????d2
graph_convolution/MatMul_1?
#graph_convolution/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d2%
#graph_convolution/Reshape_2/shape/2?
!graph_convolution/Reshape_2/shapePack"graph_convolution/unstack:output:0"graph_convolution/unstack:output:1,graph_convolution/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2#
!graph_convolution/Reshape_2/shape?
graph_convolution/Reshape_2Reshape$graph_convolution/MatMul_1:product:0*graph_convolution/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :??????????????????d2
graph_convolution/Reshape_2?
$graph_convolution/add/ReadVariableOpReadVariableOp-graph_convolution_add_readvariableop_resource*
_output_shapes
:d*
dtype02&
$graph_convolution/add/ReadVariableOp?
graph_convolution/addAddV2$graph_convolution/Reshape_2:output:0,graph_convolution/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????d2
graph_convolution/add?
graph_convolution/ReluRelugraph_convolution/add:z:0*
T0*4
_output_shapes"
 :??????????????????d2
graph_convolution/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_1/dropout/Const?
dropout_1/dropout/MulMul$graph_convolution/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*4
_output_shapes"
 :??????????????????d2
dropout_1/dropout/Mul?
dropout_1/dropout/ShapeShape$graph_convolution/Relu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*4
_output_shapes"
 :??????????????????d*
dtype0*
seed?'*
seed220
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????d2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????d2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*4
_output_shapes"
 :??????????????????d2
dropout_1/dropout/Mul_1?
graph_convolution_1/MatMulBatchMatMulV2inputs_2dropout_1/dropout/Mul_1:z:0*
T0*4
_output_shapes"
 :??????????????????d2
graph_convolution_1/MatMul?
graph_convolution_1/ShapeShape#graph_convolution_1/MatMul:output:0*
T0*
_output_shapes
:2
graph_convolution_1/Shape?
graph_convolution_1/Shape_1Shape#graph_convolution_1/MatMul:output:0*
T0*
_output_shapes
:2
graph_convolution_1/Shape_1?
graph_convolution_1/unstackUnpack$graph_convolution_1/Shape_1:output:0*
T0*
_output_shapes
: : : *	
num2
graph_convolution_1/unstack?
*graph_convolution_1/Shape_2/ReadVariableOpReadVariableOp3graph_convolution_1_shape_2_readvariableop_resource*
_output_shapes

:dd*
dtype02,
*graph_convolution_1/Shape_2/ReadVariableOp?
graph_convolution_1/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"d   d   2
graph_convolution_1/Shape_2?
graph_convolution_1/unstack_1Unpack$graph_convolution_1/Shape_2:output:0*
T0*
_output_shapes
: : *	
num2
graph_convolution_1/unstack_1?
!graph_convolution_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2#
!graph_convolution_1/Reshape/shape?
graph_convolution_1/ReshapeReshape#graph_convolution_1/MatMul:output:0*graph_convolution_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????d2
graph_convolution_1/Reshape?
,graph_convolution_1/transpose/ReadVariableOpReadVariableOp3graph_convolution_1_shape_2_readvariableop_resource*
_output_shapes

:dd*
dtype02.
,graph_convolution_1/transpose/ReadVariableOp?
"graph_convolution_1/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2$
"graph_convolution_1/transpose/perm?
graph_convolution_1/transpose	Transpose4graph_convolution_1/transpose/ReadVariableOp:value:0+graph_convolution_1/transpose/perm:output:0*
T0*
_output_shapes

:dd2
graph_convolution_1/transpose?
#graph_convolution_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"d   ????2%
#graph_convolution_1/Reshape_1/shape?
graph_convolution_1/Reshape_1Reshape!graph_convolution_1/transpose:y:0,graph_convolution_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:dd2
graph_convolution_1/Reshape_1?
graph_convolution_1/MatMul_1MatMul$graph_convolution_1/Reshape:output:0&graph_convolution_1/Reshape_1:output:0*
T0*'
_output_shapes
:?????????d2
graph_convolution_1/MatMul_1?
%graph_convolution_1/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d2'
%graph_convolution_1/Reshape_2/shape/2?
#graph_convolution_1/Reshape_2/shapePack$graph_convolution_1/unstack:output:0$graph_convolution_1/unstack:output:1.graph_convolution_1/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2%
#graph_convolution_1/Reshape_2/shape?
graph_convolution_1/Reshape_2Reshape&graph_convolution_1/MatMul_1:product:0,graph_convolution_1/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :??????????????????d2
graph_convolution_1/Reshape_2?
&graph_convolution_1/add/ReadVariableOpReadVariableOp/graph_convolution_1_add_readvariableop_resource*
_output_shapes
:d*
dtype02(
&graph_convolution_1/add/ReadVariableOp?
graph_convolution_1/addAddV2&graph_convolution_1/Reshape_2:output:0.graph_convolution_1/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????d2
graph_convolution_1/add?
graph_convolution_1/ReluRelugraph_convolution_1/add:z:0*
T0*4
_output_shapes"
 :??????????????????d2
graph_convolution_1/Relu?
,global_average_pooling1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,global_average_pooling1d/strided_slice/stack?
.global_average_pooling1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.global_average_pooling1d/strided_slice/stack_1?
.global_average_pooling1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.global_average_pooling1d/strided_slice/stack_2?
&global_average_pooling1d/strided_sliceStridedSlice&graph_convolution_1/Relu:activations:05global_average_pooling1d/strided_slice/stack:output:07global_average_pooling1d/strided_slice/stack_1:output:07global_average_pooling1d/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask2(
&global_average_pooling1d/strided_slice?
global_average_pooling1d/CastCastinputs_1*

DstT0*

SrcT0
*0
_output_shapes
:??????????????????2
global_average_pooling1d/Cast?
'global_average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'global_average_pooling1d/ExpandDims/dim?
#global_average_pooling1d/ExpandDims
ExpandDims!global_average_pooling1d/Cast:y:00global_average_pooling1d/ExpandDims/dim:output:0*
T0*4
_output_shapes"
 :??????????????????2%
#global_average_pooling1d/ExpandDims?
global_average_pooling1d/mulMul&graph_convolution_1/Relu:activations:0,global_average_pooling1d/ExpandDims:output:0*
T0*4
_output_shapes"
 :??????????????????d2
global_average_pooling1d/mul?
.global_average_pooling1d/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :20
.global_average_pooling1d/Sum/reduction_indices?
global_average_pooling1d/SumSum global_average_pooling1d/mul:z:07global_average_pooling1d/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????d2
global_average_pooling1d/Sum?
0global_average_pooling1d/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :22
0global_average_pooling1d/Sum_1/reduction_indices?
global_average_pooling1d/Sum_1Sum,global_average_pooling1d/ExpandDims:output:09global_average_pooling1d/Sum_1/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2 
global_average_pooling1d/Sum_1?
 global_average_pooling1d/truedivRealDiv%global_average_pooling1d/Sum:output:0'global_average_pooling1d/Sum_1:output:0*
T0*'
_output_shapes
:?????????d2"
 global_average_pooling1d/truedivo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????d   2
flatten/Const?
flatten/ReshapeReshape$global_average_pooling1d/truediv:z:0flatten/Const:output:0*
T0*'
_output_shapes
:?????????d2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:d@*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_1/Relu?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_2/Relu?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAddy
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3/Sigmoidn
IdentityIdentitydense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp%^graph_convolution/add/ReadVariableOp+^graph_convolution/transpose/ReadVariableOp'^graph_convolution_1/add/ReadVariableOp-^graph_convolution_1/transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes
}:??????????????????o:??????????????????:'???????????????????????????: : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2L
$graph_convolution/add/ReadVariableOp$graph_convolution/add/ReadVariableOp2X
*graph_convolution/transpose/ReadVariableOp*graph_convolution/transpose/ReadVariableOp2P
&graph_convolution_1/add/ReadVariableOp&graph_convolution_1/add/ReadVariableOp2\
,graph_convolution_1/transpose/ReadVariableOp,graph_convolution_1/transpose/ReadVariableOp:^ Z
4
_output_shapes"
 :??????????????????o
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:??????????????????
"
_user_specified_name
inputs/1:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/2
?2
?
?__inference_model_layer_call_and_return_conditional_losses_4742

inputs
inputs_1

inputs_2(
graph_convolution_4597:od$
graph_convolution_4599:d*
graph_convolution_1_4641:dd&
graph_convolution_1_4643:d

dense_4685:d@

dense_4687:@
dense_1_4702:@ 
dense_1_4704: 
dense_2_4719: 
dense_2_4721:
dense_3_4736:
dense_3_4738:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?)graph_convolution/StatefulPartitionedCall?+graph_convolution_1/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????o* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_45632
dropout/PartitionedCall?
)graph_convolution/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0inputs_2graph_convolution_4597graph_convolution_4599*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_graph_convolution_layer_call_and_return_conditional_losses_45962+
)graph_convolution/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall2graph_convolution/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_46072
dropout_1/PartitionedCall?
+graph_convolution_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0inputs_2graph_convolution_1_4641graph_convolution_1_4643*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_graph_convolution_1_layer_call_and_return_conditional_losses_46402-
+graph_convolution_1/StatefulPartitionedCall?
(global_average_pooling1d/PartitionedCallPartitionedCall4graph_convolution_1/StatefulPartitionedCall:output:0inputs_1*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_46632*
(global_average_pooling1d/PartitionedCall?
flatten/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_46712
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dense_4685
dense_4687*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_46842
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_4702dense_1_4704*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_47012!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_4719dense_2_4721*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_47182!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_4736dense_3_4738*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_47352!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*^graph_convolution/StatefulPartitionedCall,^graph_convolution_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes
}:??????????????????o:??????????????????:'???????????????????????????: : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2V
)graph_convolution/StatefulPartitionedCall)graph_convolution/StatefulPartitionedCall2Z
+graph_convolution_1/StatefulPartitionedCall+graph_convolution_1/StatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????o
 
_user_specified_nameinputs:XT
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
M__inference_graph_convolution_1_layer_call_and_return_conditional_losses_4640

inputs
inputs_11
shape_2_readvariableop_resource:dd)
add_readvariableop_resource:d
identity??add/ReadVariableOp?transpose/ReadVariableOpr
MatMulBatchMatMulV2inputs_1inputs*
T0*4
_output_shapes"
 :??????????????????d2
MatMulM
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:2
ShapeQ
Shape_1ShapeMatMul:output:0*
T0*
_output_shapes
:2	
Shape_1^
unstackUnpackShape_1:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack?
Shape_2/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

:dd*
dtype02
Shape_2/ReadVariableOpc
Shape_2Const*
_output_shapes
:*
dtype0*
valueB"d   d   2	
Shape_2`
	unstack_1UnpackShape_2:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   2
Reshape/shapex
ReshapeReshapeMatMul:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????d2	
Reshape?
transpose/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

:dd*
dtype02
transpose/ReadVariableOpq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm?
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:dd2
	transposes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"d   ????2
Reshape_1/shapes
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:dd2
	Reshape_1v
MatMul_1MatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:?????????d2

MatMul_1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d2
Reshape_2/shape/2?
Reshape_2/shapePackunstack:output:0unstack:output:1Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape?
	Reshape_2ReshapeMatMul_1:product:0Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :??????????????????d2
	Reshape_2?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:d*
dtype02
add/ReadVariableOp?
addAddV2Reshape_2:output:0add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????d2
add\
ReluReluadd:z:0*
T0*4
_output_shapes"
 :??????????????????d2
Reluz
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :??????????????????d2

Identity~
NoOpNoOp^add/ReadVariableOp^transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:??????????????????d:'???????????????????????????: : 2(
add/ReadVariableOpadd/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????d
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
H
input_1=
serving_default_input_1:0??????????????????o
D
input_29
serving_default_input_2:0
??????????????????
Q
input_3F
serving_default_input_3:0'???????????????????????????;
dense_30
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-2

layer-9
layer_with_weights-3
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
?
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
"
_tf_keras_input_layer
?

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
trainable_variables
regularization_losses
 	variables
!	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

"kernel
#bias
$trainable_variables
%regularization_losses
&	variables
'	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
"
_tf_keras_input_layer
?
(trainable_variables
)regularization_losses
*	variables
+	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
,trainable_variables
-regularization_losses
.	variables
/	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

0kernel
1bias
2trainable_variables
3regularization_losses
4	variables
5	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

6kernel
7bias
8trainable_variables
9regularization_losses
:	variables
;	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

<kernel
=bias
>trainable_variables
?regularization_losses
@	variables
A	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

Bkernel
Cbias
Dtrainable_variables
Eregularization_losses
F	variables
G	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
Hiter

Ibeta_1

Jbeta_2
	Kdecay
Llearning_ratem?m?"m?#m?0m?1m?6m?7m?<m?=m?Bm?Cm?v?v?"v?#v?0v?1v?6v?7v?<v?=v?Bv?Cv?"
	optimizer
v
0
1
"2
#3
04
15
66
77
<8
=9
B10
C11"
trackable_list_wrapper
 "
trackable_list_wrapper
v
0
1
"2
#3
04
15
66
77
<8
=9
B10
C11"
trackable_list_wrapper
?
Mnon_trainable_variables

Nlayers
Olayer_metrics
trainable_variables
Pmetrics
regularization_losses
	variables
Qlayer_regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Rnon_trainable_variables
Slayer_metrics

Tlayers
Umetrics
trainable_variables
regularization_losses
	variables
Vlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(od2graph_convolution/kernel
$:"d2graph_convolution/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Wnon_trainable_variables
Xlayer_metrics

Ylayers
Zmetrics
trainable_variables
regularization_losses
	variables
[layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
\non_trainable_variables
]layer_metrics

^layers
_metrics
trainable_variables
regularization_losses
 	variables
`layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*dd2graph_convolution_1/kernel
&:$d2graph_convolution_1/bias
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
?
anon_trainable_variables
blayer_metrics

clayers
dmetrics
$trainable_variables
%regularization_losses
&	variables
elayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
fnon_trainable_variables
glayer_metrics

hlayers
imetrics
(trainable_variables
)regularization_losses
*	variables
jlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
knon_trainable_variables
llayer_metrics

mlayers
nmetrics
,trainable_variables
-regularization_losses
.	variables
olayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:d@2dense/kernel
:@2
dense/bias
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
?
pnon_trainable_variables
qlayer_metrics

rlayers
smetrics
2trainable_variables
3regularization_losses
4	variables
tlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :@ 2dense_1/kernel
: 2dense_1/bias
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
?
unon_trainable_variables
vlayer_metrics

wlayers
xmetrics
8trainable_variables
9regularization_losses
:	variables
ylayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 : 2dense_2/kernel
:2dense_2/bias
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
?
znon_trainable_variables
{layer_metrics

|layers
}metrics
>trainable_variables
?regularization_losses
@	variables
~layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :2dense_3/kernel
:2dense_3/bias
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
?
non_trainable_variables
?layer_metrics
?layers
?metrics
Dtrainable_variables
Eregularization_losses
F	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
	8

9
10
11
12"
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
/:-od2Adam/graph_convolution/kernel/m
):'d2Adam/graph_convolution/bias/m
1:/dd2!Adam/graph_convolution_1/kernel/m
+:)d2Adam/graph_convolution_1/bias/m
#:!d@2Adam/dense/kernel/m
:@2Adam/dense/bias/m
%:#@ 2Adam/dense_1/kernel/m
: 2Adam/dense_1/bias/m
%:# 2Adam/dense_2/kernel/m
:2Adam/dense_2/bias/m
%:#2Adam/dense_3/kernel/m
:2Adam/dense_3/bias/m
/:-od2Adam/graph_convolution/kernel/v
):'d2Adam/graph_convolution/bias/v
1:/dd2!Adam/graph_convolution_1/kernel/v
+:)d2Adam/graph_convolution_1/bias/v
#:!d@2Adam/dense/kernel/v
:@2Adam/dense/bias/v
%:#@ 2Adam/dense_1/kernel/v
: 2Adam/dense_1/bias/v
%:# 2Adam/dense_2/kernel/v
:2Adam/dense_2/bias/v
%:#2Adam/dense_3/kernel/v
:2Adam/dense_3/bias/v
?2?
?__inference_model_layer_call_and_return_conditional_losses_5244
?__inference_model_layer_call_and_return_conditional_losses_5361
?__inference_model_layer_call_and_return_conditional_losses_5062
?__inference_model_layer_call_and_return_conditional_losses_5102?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
$__inference_model_layer_call_fn_4769
$__inference_model_layer_call_fn_5392
$__inference_model_layer_call_fn_5423
$__inference_model_layer_call_fn_5022?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
__inference__wrapped_model_4523input_1input_2input_3"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dropout_layer_call_and_return_conditional_losses_5428
A__inference_dropout_layer_call_and_return_conditional_losses_5440?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_dropout_layer_call_fn_5445
&__inference_dropout_layer_call_fn_5450?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
K__inference_graph_convolution_layer_call_and_return_conditional_losses_5481?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_graph_convolution_layer_call_fn_5491?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dropout_1_layer_call_and_return_conditional_losses_5496
C__inference_dropout_1_layer_call_and_return_conditional_losses_5508?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dropout_1_layer_call_fn_5513
(__inference_dropout_1_layer_call_fn_5518?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
M__inference_graph_convolution_1_layer_call_and_return_conditional_losses_5549?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_graph_convolution_1_layer_call_fn_5559?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_5565
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_5583?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_global_average_pooling1d_layer_call_fn_5588
7__inference_global_average_pooling1d_layer_call_fn_5594?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_flatten_layer_call_and_return_conditional_losses_5600?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_flatten_layer_call_fn_5605?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_dense_layer_call_and_return_conditional_losses_5616?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_dense_layer_call_fn_5625?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_1_layer_call_and_return_conditional_losses_5636?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_1_layer_call_fn_5645?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_2_layer_call_and_return_conditional_losses_5656?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_2_layer_call_fn_5665?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_3_layer_call_and_return_conditional_losses_5676?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_3_layer_call_fn_5685?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
"__inference_signature_wrapper_5141input_1input_2input_3"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
__inference__wrapped_model_4523?"#0167<=BC???
???
???
.?+
input_1??????????????????o
*?'
input_2??????????????????

7?4
input_3'???????????????????????????
? "1?.
,
dense_3!?
dense_3??????????
A__inference_dense_1_layer_call_and_return_conditional_losses_5636\67/?,
%?"
 ?
inputs?????????@
? "%?"
?
0????????? 
? y
&__inference_dense_1_layer_call_fn_5645O67/?,
%?"
 ?
inputs?????????@
? "?????????? ?
A__inference_dense_2_layer_call_and_return_conditional_losses_5656\<=/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? y
&__inference_dense_2_layer_call_fn_5665O<=/?,
%?"
 ?
inputs????????? 
? "???????????
A__inference_dense_3_layer_call_and_return_conditional_losses_5676\BC/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? y
&__inference_dense_3_layer_call_fn_5685OBC/?,
%?"
 ?
inputs?????????
? "???????????
?__inference_dense_layer_call_and_return_conditional_losses_5616\01/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????@
? w
$__inference_dense_layer_call_fn_5625O01/?,
%?"
 ?
inputs?????????d
? "??????????@?
C__inference_dropout_1_layer_call_and_return_conditional_losses_5496v@?=
6?3
-?*
inputs??????????????????d
p 
? "2?/
(?%
0??????????????????d
? ?
C__inference_dropout_1_layer_call_and_return_conditional_losses_5508v@?=
6?3
-?*
inputs??????????????????d
p
? "2?/
(?%
0??????????????????d
? ?
(__inference_dropout_1_layer_call_fn_5513i@?=
6?3
-?*
inputs??????????????????d
p 
? "%?"??????????????????d?
(__inference_dropout_1_layer_call_fn_5518i@?=
6?3
-?*
inputs??????????????????d
p
? "%?"??????????????????d?
A__inference_dropout_layer_call_and_return_conditional_losses_5428v@?=
6?3
-?*
inputs??????????????????o
p 
? "2?/
(?%
0??????????????????o
? ?
A__inference_dropout_layer_call_and_return_conditional_losses_5440v@?=
6?3
-?*
inputs??????????????????o
p
? "2?/
(?%
0??????????????????o
? ?
&__inference_dropout_layer_call_fn_5445i@?=
6?3
-?*
inputs??????????????????o
p 
? "%?"??????????????????o?
&__inference_dropout_layer_call_fn_5450i@?=
6?3
-?*
inputs??????????????????o
p
? "%?"??????????????????o?
A__inference_flatten_layer_call_and_return_conditional_losses_5600X/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????d
? u
&__inference_flatten_layer_call_fn_5605K/?,
%?"
 ?
inputs?????????d
? "??????????d?
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_5565{I?F
??<
6?3
inputs'???????????????????????????

 
? ".?+
$?!
0??????????????????
? ?
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_5583?e?b
[?X
-?*
inputs??????????????????d
'?$
mask??????????????????

? "%?"
?
0?????????d
? ?
7__inference_global_average_pooling1d_layer_call_fn_5588nI?F
??<
6?3
inputs'???????????????????????????

 
? "!????????????????????
7__inference_global_average_pooling1d_layer_call_fn_5594?e?b
[?X
-?*
inputs??????????????????d
'?$
mask??????????????????

? "??????????d?
M__inference_graph_convolution_1_layer_call_and_return_conditional_losses_5549?"#}?z
s?p
n?k
/?,
inputs/0??????????????????d
8?5
inputs/1'???????????????????????????
? "2?/
(?%
0??????????????????d
? ?
2__inference_graph_convolution_1_layer_call_fn_5559?"#}?z
s?p
n?k
/?,
inputs/0??????????????????d
8?5
inputs/1'???????????????????????????
? "%?"??????????????????d?
K__inference_graph_convolution_layer_call_and_return_conditional_losses_5481?}?z
s?p
n?k
/?,
inputs/0??????????????????o
8?5
inputs/1'???????????????????????????
? "2?/
(?%
0??????????????????d
? ?
0__inference_graph_convolution_layer_call_fn_5491?}?z
s?p
n?k
/?,
inputs/0??????????????????o
8?5
inputs/1'???????????????????????????
? "%?"??????????????????d?
?__inference_model_layer_call_and_return_conditional_losses_5062?"#0167<=BC???
???
???
.?+
input_1??????????????????o
*?'
input_2??????????????????

7?4
input_3'???????????????????????????
p 

 
? "%?"
?
0?????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_5102?"#0167<=BC???
???
???
.?+
input_1??????????????????o
*?'
input_2??????????????????

7?4
input_3'???????????????????????????
p

 
? "%?"
?
0?????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_5244?"#0167<=BC???
???
???
/?,
inputs/0??????????????????o
+?(
inputs/1??????????????????

8?5
inputs/2'???????????????????????????
p 

 
? "%?"
?
0?????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_5361?"#0167<=BC???
???
???
/?,
inputs/0??????????????????o
+?(
inputs/1??????????????????

8?5
inputs/2'???????????????????????????
p

 
? "%?"
?
0?????????
? ?
$__inference_model_layer_call_fn_4769?"#0167<=BC???
???
???
.?+
input_1??????????????????o
*?'
input_2??????????????????

7?4
input_3'???????????????????????????
p 

 
? "???????????
$__inference_model_layer_call_fn_5022?"#0167<=BC???
???
???
.?+
input_1??????????????????o
*?'
input_2??????????????????

7?4
input_3'???????????????????????????
p

 
? "???????????
$__inference_model_layer_call_fn_5392?"#0167<=BC???
???
???
/?,
inputs/0??????????????????o
+?(
inputs/1??????????????????

8?5
inputs/2'???????????????????????????
p 

 
? "???????????
$__inference_model_layer_call_fn_5423?"#0167<=BC???
???
???
/?,
inputs/0??????????????????o
+?(
inputs/1??????????????????

8?5
inputs/2'???????????????????????????
p

 
? "???????????
"__inference_signature_wrapper_5141?"#0167<=BC???
? 
???
9
input_1.?+
input_1??????????????????o
5
input_2*?'
input_2??????????????????

B
input_37?4
input_3'???????????????????????????"1?.
,
dense_3!?
dense_3?????????