�
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

�
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
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
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.13.02v2.13.0-rc2-7-g1cb1a030a628��	
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
�
Adam/v/conv2d_transpose_65/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/v/conv2d_transpose_65/bias
�
3Adam/v/conv2d_transpose_65/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_transpose_65/bias*
_output_shapes
:*
dtype0
�
Adam/m/conv2d_transpose_65/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/m/conv2d_transpose_65/bias
�
3Adam/m/conv2d_transpose_65/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_transpose_65/bias*
_output_shapes
:*
dtype0
�
!Adam/v/conv2d_transpose_65/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/v/conv2d_transpose_65/kernel
�
5Adam/v/conv2d_transpose_65/kernel/Read/ReadVariableOpReadVariableOp!Adam/v/conv2d_transpose_65/kernel*&
_output_shapes
: *
dtype0
�
!Adam/m/conv2d_transpose_65/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/m/conv2d_transpose_65/kernel
�
5Adam/m/conv2d_transpose_65/kernel/Read/ReadVariableOpReadVariableOp!Adam/m/conv2d_transpose_65/kernel*&
_output_shapes
: *
dtype0
�
Adam/v/conv2d_transpose_64/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/v/conv2d_transpose_64/bias
�
3Adam/v/conv2d_transpose_64/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_transpose_64/bias*
_output_shapes
: *
dtype0
�
Adam/m/conv2d_transpose_64/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/m/conv2d_transpose_64/bias
�
3Adam/m/conv2d_transpose_64/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_transpose_64/bias*
_output_shapes
: *
dtype0
�
!Adam/v/conv2d_transpose_64/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/v/conv2d_transpose_64/kernel
�
5Adam/v/conv2d_transpose_64/kernel/Read/ReadVariableOpReadVariableOp!Adam/v/conv2d_transpose_64/kernel*&
_output_shapes
: *
dtype0
�
!Adam/m/conv2d_transpose_64/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/m/conv2d_transpose_64/kernel
�
5Adam/m/conv2d_transpose_64/kernel/Read/ReadVariableOpReadVariableOp!Adam/m/conv2d_transpose_64/kernel*&
_output_shapes
: *
dtype0
�
Adam/v/conv2d_transpose_63/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/v/conv2d_transpose_63/bias
�
3Adam/v/conv2d_transpose_63/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_transpose_63/bias*
_output_shapes
:*
dtype0
�
Adam/m/conv2d_transpose_63/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/m/conv2d_transpose_63/bias
�
3Adam/m/conv2d_transpose_63/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_transpose_63/bias*
_output_shapes
:*
dtype0
�
!Adam/v/conv2d_transpose_63/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/v/conv2d_transpose_63/kernel
�
5Adam/v/conv2d_transpose_63/kernel/Read/ReadVariableOpReadVariableOp!Adam/v/conv2d_transpose_63/kernel*&
_output_shapes
:*
dtype0
�
!Adam/m/conv2d_transpose_63/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/m/conv2d_transpose_63/kernel
�
5Adam/m/conv2d_transpose_63/kernel/Read/ReadVariableOpReadVariableOp!Adam/m/conv2d_transpose_63/kernel*&
_output_shapes
:*
dtype0
�
Adam/v/conv2d_43/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/v/conv2d_43/bias
{
)Adam/v/conv2d_43/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_43/bias*
_output_shapes
:*
dtype0
�
Adam/m/conv2d_43/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/m/conv2d_43/bias
{
)Adam/m/conv2d_43/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_43/bias*
_output_shapes
:*
dtype0
�
Adam/v/conv2d_43/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/v/conv2d_43/kernel
�
+Adam/v/conv2d_43/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_43/kernel*&
_output_shapes
: *
dtype0
�
Adam/m/conv2d_43/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/m/conv2d_43/kernel
�
+Adam/m/conv2d_43/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_43/kernel*&
_output_shapes
: *
dtype0
�
Adam/v/conv2d_42/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/v/conv2d_42/bias
{
)Adam/v/conv2d_42/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_42/bias*
_output_shapes
: *
dtype0
�
Adam/m/conv2d_42/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/m/conv2d_42/bias
{
)Adam/m/conv2d_42/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_42/bias*
_output_shapes
: *
dtype0
�
Adam/v/conv2d_42/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/v/conv2d_42/kernel
�
+Adam/v/conv2d_42/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_42/kernel*&
_output_shapes
: *
dtype0
�
Adam/m/conv2d_42/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/m/conv2d_42/kernel
�
+Adam/m/conv2d_42/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_42/kernel*&
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
conv2d_transpose_65/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_65/bias
�
,conv2d_transpose_65/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_65/bias*
_output_shapes
:*
dtype0
�
conv2d_transpose_65/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameconv2d_transpose_65/kernel
�
.conv2d_transpose_65/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_65/kernel*&
_output_shapes
: *
dtype0
�
conv2d_transpose_64/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameconv2d_transpose_64/bias
�
,conv2d_transpose_64/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_64/bias*
_output_shapes
: *
dtype0
�
conv2d_transpose_64/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameconv2d_transpose_64/kernel
�
.conv2d_transpose_64/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_64/kernel*&
_output_shapes
: *
dtype0
�
conv2d_transpose_63/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_63/bias
�
,conv2d_transpose_63/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_63/bias*
_output_shapes
:*
dtype0
�
conv2d_transpose_63/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameconv2d_transpose_63/kernel
�
.conv2d_transpose_63/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_63/kernel*&
_output_shapes
:*
dtype0
t
conv2d_43/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_43/bias
m
"conv2d_43/bias/Read/ReadVariableOpReadVariableOpconv2d_43/bias*
_output_shapes
:*
dtype0
�
conv2d_43/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_43/kernel
}
$conv2d_43/kernel/Read/ReadVariableOpReadVariableOpconv2d_43/kernel*&
_output_shapes
: *
dtype0
t
conv2d_42/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_42/bias
m
"conv2d_42/bias/Read/ReadVariableOpReadVariableOpconv2d_42/bias*
_output_shapes
: *
dtype0
�
conv2d_42/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_42/kernel
}
$conv2d_42/kernel/Read/ReadVariableOpReadVariableOpconv2d_42/kernel*&
_output_shapes
: *
dtype0
�
serving_default_input_30Placeholder*/
_output_shapes
:���������*
dtype0*$
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_30conv2d_42/kernelconv2d_42/biasconv2d_43/kernelconv2d_43/biasconv2d_transpose_63/kernelconv2d_transpose_63/biasconv2d_transpose_64/kernelconv2d_transpose_64/biasconv2d_transpose_65/kernelconv2d_transpose_65/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_3492488

NoOpNoOp
�H
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�H
value�HB�H B�H
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
 _random_generator* 
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses

'kernel
(bias
 )_jit_compiled_convolution_op*
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses

0kernel
1bias
 2_jit_compiled_convolution_op*
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses
9_random_generator* 
�
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

@kernel
Abias
 B_jit_compiled_convolution_op*
�
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses

Ikernel
Jbias
 K_jit_compiled_convolution_op*
J
0
1
'2
(3
04
15
@6
A7
I8
J9*
J
0
1
'2
(3
04
15
@6
A7
I8
J9*
* 
�
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Qtrace_0
Rtrace_1* 

Strace_0
Ttrace_1* 
* 
�
U
_variables
V_iterations
W_learning_rate
X_index_dict
Y
_momentums
Z_velocities
[_update_step_xla*

\serving_default* 

0
1*

0
1*
* 
�
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

btrace_0* 

ctrace_0* 
`Z
VARIABLE_VALUEconv2d_42/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_42/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

itrace_0
jtrace_1* 

ktrace_0
ltrace_1* 
* 

'0
(1*

'0
(1*
* 
�
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*

rtrace_0* 

strace_0* 
`Z
VARIABLE_VALUEconv2d_43/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_43/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

00
11*

00
11*
* 
�
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*

ytrace_0* 

ztrace_0* 
jd
VARIABLE_VALUEconv2d_transpose_63/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEconv2d_transpose_63/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

@0
A1*

@0
A1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
jd
VARIABLE_VALUEconv2d_transpose_64/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEconv2d_transpose_64/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

I0
J1*

I0
J1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
jd
VARIABLE_VALUEconv2d_transpose_65/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEconv2d_transpose_65/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
5
0
1
2
3
4
5
6*

�0*
* 
* 
* 
* 
* 
* 
�
V0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
T
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9*
T
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
b\
VARIABLE_VALUEAdam/m/conv2d_42/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_42/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_42/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_42/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_43/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_43/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_43/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_43/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE!Adam/m/conv2d_transpose_63/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/conv2d_transpose_63/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/m/conv2d_transpose_63/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/v/conv2d_transpose_63/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/conv2d_transpose_64/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/conv2d_transpose_64/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/m/conv2d_transpose_64/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/v/conv2d_transpose_64/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/conv2d_transpose_65/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/conv2d_transpose_65/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/m/conv2d_transpose_65/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/v/conv2d_transpose_65/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv2d_42/kernelconv2d_42/biasconv2d_43/kernelconv2d_43/biasconv2d_transpose_63/kernelconv2d_transpose_63/biasconv2d_transpose_64/kernelconv2d_transpose_64/biasconv2d_transpose_65/kernelconv2d_transpose_65/bias	iterationlearning_rateAdam/m/conv2d_42/kernelAdam/v/conv2d_42/kernelAdam/m/conv2d_42/biasAdam/v/conv2d_42/biasAdam/m/conv2d_43/kernelAdam/v/conv2d_43/kernelAdam/m/conv2d_43/biasAdam/v/conv2d_43/bias!Adam/m/conv2d_transpose_63/kernel!Adam/v/conv2d_transpose_63/kernelAdam/m/conv2d_transpose_63/biasAdam/v/conv2d_transpose_63/bias!Adam/m/conv2d_transpose_64/kernel!Adam/v/conv2d_transpose_64/kernelAdam/m/conv2d_transpose_64/biasAdam/v/conv2d_transpose_64/bias!Adam/m/conv2d_transpose_65/kernel!Adam/v/conv2d_transpose_65/kernelAdam/m/conv2d_transpose_65/biasAdam/v/conv2d_transpose_65/biastotalcountConst*/
Tin(
&2$*
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
GPU 2J 8� *)
f$R"
 __inference__traced_save_3492936
�	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_42/kernelconv2d_42/biasconv2d_43/kernelconv2d_43/biasconv2d_transpose_63/kernelconv2d_transpose_63/biasconv2d_transpose_64/kernelconv2d_transpose_64/biasconv2d_transpose_65/kernelconv2d_transpose_65/bias	iterationlearning_rateAdam/m/conv2d_42/kernelAdam/v/conv2d_42/kernelAdam/m/conv2d_42/biasAdam/v/conv2d_42/biasAdam/m/conv2d_43/kernelAdam/v/conv2d_43/kernelAdam/m/conv2d_43/biasAdam/v/conv2d_43/bias!Adam/m/conv2d_transpose_63/kernel!Adam/v/conv2d_transpose_63/kernelAdam/m/conv2d_transpose_63/biasAdam/v/conv2d_transpose_63/bias!Adam/m/conv2d_transpose_64/kernel!Adam/v/conv2d_transpose_64/kernelAdam/m/conv2d_transpose_64/biasAdam/v/conv2d_transpose_64/bias!Adam/m/conv2d_transpose_65/kernel!Adam/v/conv2d_transpose_65/kernelAdam/m/conv2d_transpose_65/biasAdam/v/conv2d_transpose_65/biastotalcount*.
Tin'
%2#*
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
GPU 2J 8� *,
f'R%
#__inference__traced_restore_3493047��
�
e
G__inference_dropout_58_layer_call_and_return_conditional_losses_3492343

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:��������� c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�!
�
P__inference_conv2d_transpose_65_layer_call_and_return_conditional_losses_3492710

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������]
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
/__inference_sequential_28_layer_call_fn_3492397
input_30!
unknown: 
	unknown_0: #
	unknown_1: 
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: #
	unknown_7: 
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_30unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_28_layer_call_and_return_conditional_losses_3492331w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'
#
!
_user_specified_name	3492393:'	#
!
_user_specified_name	3492391:'#
!
_user_specified_name	3492389:'#
!
_user_specified_name	3492387:'#
!
_user_specified_name	3492385:'#
!
_user_specified_name	3492383:'#
!
_user_specified_name	3492381:'#
!
_user_specified_name	3492379:'#
!
_user_specified_name	3492377:'#
!
_user_specified_name	3492375:Y U
/
_output_shapes
:���������
"
_user_specified_name
input_30
��
�
"__inference__wrapped_model_3492126
input_30P
6sequential_28_conv2d_42_conv2d_readvariableop_resource: E
7sequential_28_conv2d_42_biasadd_readvariableop_resource: P
6sequential_28_conv2d_43_conv2d_readvariableop_resource: E
7sequential_28_conv2d_43_biasadd_readvariableop_resource:d
Jsequential_28_conv2d_transpose_63_conv2d_transpose_readvariableop_resource:O
Asequential_28_conv2d_transpose_63_biasadd_readvariableop_resource:d
Jsequential_28_conv2d_transpose_64_conv2d_transpose_readvariableop_resource: O
Asequential_28_conv2d_transpose_64_biasadd_readvariableop_resource: d
Jsequential_28_conv2d_transpose_65_conv2d_transpose_readvariableop_resource: O
Asequential_28_conv2d_transpose_65_biasadd_readvariableop_resource:
identity��.sequential_28/conv2d_42/BiasAdd/ReadVariableOp�-sequential_28/conv2d_42/Conv2D/ReadVariableOp�.sequential_28/conv2d_43/BiasAdd/ReadVariableOp�-sequential_28/conv2d_43/Conv2D/ReadVariableOp�8sequential_28/conv2d_transpose_63/BiasAdd/ReadVariableOp�Asequential_28/conv2d_transpose_63/conv2d_transpose/ReadVariableOp�8sequential_28/conv2d_transpose_64/BiasAdd/ReadVariableOp�Asequential_28/conv2d_transpose_64/conv2d_transpose/ReadVariableOp�8sequential_28/conv2d_transpose_65/BiasAdd/ReadVariableOp�Asequential_28/conv2d_transpose_65/conv2d_transpose/ReadVariableOp�
-sequential_28/conv2d_42/Conv2D/ReadVariableOpReadVariableOp6sequential_28_conv2d_42_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
sequential_28/conv2d_42/Conv2DConv2Dinput_305sequential_28/conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
.sequential_28/conv2d_42/BiasAdd/ReadVariableOpReadVariableOp7sequential_28_conv2d_42_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
sequential_28/conv2d_42/BiasAddBiasAdd'sequential_28/conv2d_42/Conv2D:output:06sequential_28/conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
sequential_28/conv2d_42/ReluRelu(sequential_28/conv2d_42/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
!sequential_28/dropout_58/IdentityIdentity*sequential_28/conv2d_42/Relu:activations:0*
T0*/
_output_shapes
:��������� �
-sequential_28/conv2d_43/Conv2D/ReadVariableOpReadVariableOp6sequential_28_conv2d_43_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
sequential_28/conv2d_43/Conv2DConv2D*sequential_28/dropout_58/Identity:output:05sequential_28/conv2d_43/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
.sequential_28/conv2d_43/BiasAdd/ReadVariableOpReadVariableOp7sequential_28_conv2d_43_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_28/conv2d_43/BiasAddBiasAdd'sequential_28/conv2d_43/Conv2D:output:06sequential_28/conv2d_43/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
sequential_28/conv2d_43/ReluRelu(sequential_28/conv2d_43/BiasAdd:output:0*
T0*/
_output_shapes
:����������
'sequential_28/conv2d_transpose_63/ShapeShape*sequential_28/conv2d_43/Relu:activations:0*
T0*
_output_shapes
::��
5sequential_28/conv2d_transpose_63/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
7sequential_28/conv2d_transpose_63/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
7sequential_28/conv2d_transpose_63/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
/sequential_28/conv2d_transpose_63/strided_sliceStridedSlice0sequential_28/conv2d_transpose_63/Shape:output:0>sequential_28/conv2d_transpose_63/strided_slice/stack:output:0@sequential_28/conv2d_transpose_63/strided_slice/stack_1:output:0@sequential_28/conv2d_transpose_63/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)sequential_28/conv2d_transpose_63/stack/1Const*
_output_shapes
: *
dtype0*
value	B :k
)sequential_28/conv2d_transpose_63/stack/2Const*
_output_shapes
: *
dtype0*
value	B :k
)sequential_28/conv2d_transpose_63/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
'sequential_28/conv2d_transpose_63/stackPack8sequential_28/conv2d_transpose_63/strided_slice:output:02sequential_28/conv2d_transpose_63/stack/1:output:02sequential_28/conv2d_transpose_63/stack/2:output:02sequential_28/conv2d_transpose_63/stack/3:output:0*
N*
T0*
_output_shapes
:�
7sequential_28/conv2d_transpose_63/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
9sequential_28/conv2d_transpose_63/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
9sequential_28/conv2d_transpose_63/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
1sequential_28/conv2d_transpose_63/strided_slice_1StridedSlice0sequential_28/conv2d_transpose_63/stack:output:0@sequential_28/conv2d_transpose_63/strided_slice_1/stack:output:0Bsequential_28/conv2d_transpose_63/strided_slice_1/stack_1:output:0Bsequential_28/conv2d_transpose_63/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Asequential_28/conv2d_transpose_63/conv2d_transpose/ReadVariableOpReadVariableOpJsequential_28_conv2d_transpose_63_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0�
2sequential_28/conv2d_transpose_63/conv2d_transposeConv2DBackpropInput0sequential_28/conv2d_transpose_63/stack:output:0Isequential_28/conv2d_transpose_63/conv2d_transpose/ReadVariableOp:value:0*sequential_28/conv2d_43/Relu:activations:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
8sequential_28/conv2d_transpose_63/BiasAdd/ReadVariableOpReadVariableOpAsequential_28_conv2d_transpose_63_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
)sequential_28/conv2d_transpose_63/BiasAddBiasAdd;sequential_28/conv2d_transpose_63/conv2d_transpose:output:0@sequential_28/conv2d_transpose_63/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
&sequential_28/conv2d_transpose_63/ReluRelu2sequential_28/conv2d_transpose_63/BiasAdd:output:0*
T0*/
_output_shapes
:����������
!sequential_28/dropout_59/IdentityIdentity4sequential_28/conv2d_transpose_63/Relu:activations:0*
T0*/
_output_shapes
:����������
'sequential_28/conv2d_transpose_64/ShapeShape*sequential_28/dropout_59/Identity:output:0*
T0*
_output_shapes
::��
5sequential_28/conv2d_transpose_64/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
7sequential_28/conv2d_transpose_64/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
7sequential_28/conv2d_transpose_64/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
/sequential_28/conv2d_transpose_64/strided_sliceStridedSlice0sequential_28/conv2d_transpose_64/Shape:output:0>sequential_28/conv2d_transpose_64/strided_slice/stack:output:0@sequential_28/conv2d_transpose_64/strided_slice/stack_1:output:0@sequential_28/conv2d_transpose_64/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)sequential_28/conv2d_transpose_64/stack/1Const*
_output_shapes
: *
dtype0*
value	B :k
)sequential_28/conv2d_transpose_64/stack/2Const*
_output_shapes
: *
dtype0*
value	B :k
)sequential_28/conv2d_transpose_64/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
'sequential_28/conv2d_transpose_64/stackPack8sequential_28/conv2d_transpose_64/strided_slice:output:02sequential_28/conv2d_transpose_64/stack/1:output:02sequential_28/conv2d_transpose_64/stack/2:output:02sequential_28/conv2d_transpose_64/stack/3:output:0*
N*
T0*
_output_shapes
:�
7sequential_28/conv2d_transpose_64/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
9sequential_28/conv2d_transpose_64/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
9sequential_28/conv2d_transpose_64/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
1sequential_28/conv2d_transpose_64/strided_slice_1StridedSlice0sequential_28/conv2d_transpose_64/stack:output:0@sequential_28/conv2d_transpose_64/strided_slice_1/stack:output:0Bsequential_28/conv2d_transpose_64/strided_slice_1/stack_1:output:0Bsequential_28/conv2d_transpose_64/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Asequential_28/conv2d_transpose_64/conv2d_transpose/ReadVariableOpReadVariableOpJsequential_28_conv2d_transpose_64_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
2sequential_28/conv2d_transpose_64/conv2d_transposeConv2DBackpropInput0sequential_28/conv2d_transpose_64/stack:output:0Isequential_28/conv2d_transpose_64/conv2d_transpose/ReadVariableOp:value:0*sequential_28/dropout_59/Identity:output:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
8sequential_28/conv2d_transpose_64/BiasAdd/ReadVariableOpReadVariableOpAsequential_28_conv2d_transpose_64_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
)sequential_28/conv2d_transpose_64/BiasAddBiasAdd;sequential_28/conv2d_transpose_64/conv2d_transpose:output:0@sequential_28/conv2d_transpose_64/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
&sequential_28/conv2d_transpose_64/ReluRelu2sequential_28/conv2d_transpose_64/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
'sequential_28/conv2d_transpose_65/ShapeShape4sequential_28/conv2d_transpose_64/Relu:activations:0*
T0*
_output_shapes
::��
5sequential_28/conv2d_transpose_65/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
7sequential_28/conv2d_transpose_65/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
7sequential_28/conv2d_transpose_65/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
/sequential_28/conv2d_transpose_65/strided_sliceStridedSlice0sequential_28/conv2d_transpose_65/Shape:output:0>sequential_28/conv2d_transpose_65/strided_slice/stack:output:0@sequential_28/conv2d_transpose_65/strided_slice/stack_1:output:0@sequential_28/conv2d_transpose_65/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)sequential_28/conv2d_transpose_65/stack/1Const*
_output_shapes
: *
dtype0*
value	B :k
)sequential_28/conv2d_transpose_65/stack/2Const*
_output_shapes
: *
dtype0*
value	B :k
)sequential_28/conv2d_transpose_65/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
'sequential_28/conv2d_transpose_65/stackPack8sequential_28/conv2d_transpose_65/strided_slice:output:02sequential_28/conv2d_transpose_65/stack/1:output:02sequential_28/conv2d_transpose_65/stack/2:output:02sequential_28/conv2d_transpose_65/stack/3:output:0*
N*
T0*
_output_shapes
:�
7sequential_28/conv2d_transpose_65/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
9sequential_28/conv2d_transpose_65/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
9sequential_28/conv2d_transpose_65/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
1sequential_28/conv2d_transpose_65/strided_slice_1StridedSlice0sequential_28/conv2d_transpose_65/stack:output:0@sequential_28/conv2d_transpose_65/strided_slice_1/stack:output:0Bsequential_28/conv2d_transpose_65/strided_slice_1/stack_1:output:0Bsequential_28/conv2d_transpose_65/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Asequential_28/conv2d_transpose_65/conv2d_transpose/ReadVariableOpReadVariableOpJsequential_28_conv2d_transpose_65_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
2sequential_28/conv2d_transpose_65/conv2d_transposeConv2DBackpropInput0sequential_28/conv2d_transpose_65/stack:output:0Isequential_28/conv2d_transpose_65/conv2d_transpose/ReadVariableOp:value:04sequential_28/conv2d_transpose_64/Relu:activations:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
8sequential_28/conv2d_transpose_65/BiasAdd/ReadVariableOpReadVariableOpAsequential_28_conv2d_transpose_65_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
)sequential_28/conv2d_transpose_65/BiasAddBiasAdd;sequential_28/conv2d_transpose_65/conv2d_transpose:output:0@sequential_28/conv2d_transpose_65/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
IdentityIdentity2sequential_28/conv2d_transpose_65/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp/^sequential_28/conv2d_42/BiasAdd/ReadVariableOp.^sequential_28/conv2d_42/Conv2D/ReadVariableOp/^sequential_28/conv2d_43/BiasAdd/ReadVariableOp.^sequential_28/conv2d_43/Conv2D/ReadVariableOp9^sequential_28/conv2d_transpose_63/BiasAdd/ReadVariableOpB^sequential_28/conv2d_transpose_63/conv2d_transpose/ReadVariableOp9^sequential_28/conv2d_transpose_64/BiasAdd/ReadVariableOpB^sequential_28/conv2d_transpose_64/conv2d_transpose/ReadVariableOp9^sequential_28/conv2d_transpose_65/BiasAdd/ReadVariableOpB^sequential_28/conv2d_transpose_65/conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : 2`
.sequential_28/conv2d_42/BiasAdd/ReadVariableOp.sequential_28/conv2d_42/BiasAdd/ReadVariableOp2^
-sequential_28/conv2d_42/Conv2D/ReadVariableOp-sequential_28/conv2d_42/Conv2D/ReadVariableOp2`
.sequential_28/conv2d_43/BiasAdd/ReadVariableOp.sequential_28/conv2d_43/BiasAdd/ReadVariableOp2^
-sequential_28/conv2d_43/Conv2D/ReadVariableOp-sequential_28/conv2d_43/Conv2D/ReadVariableOp2t
8sequential_28/conv2d_transpose_63/BiasAdd/ReadVariableOp8sequential_28/conv2d_transpose_63/BiasAdd/ReadVariableOp2�
Asequential_28/conv2d_transpose_63/conv2d_transpose/ReadVariableOpAsequential_28/conv2d_transpose_63/conv2d_transpose/ReadVariableOp2t
8sequential_28/conv2d_transpose_64/BiasAdd/ReadVariableOp8sequential_28/conv2d_transpose_64/BiasAdd/ReadVariableOp2�
Asequential_28/conv2d_transpose_64/conv2d_transpose/ReadVariableOpAsequential_28/conv2d_transpose_64/conv2d_transpose/ReadVariableOp2t
8sequential_28/conv2d_transpose_65/BiasAdd/ReadVariableOp8sequential_28/conv2d_transpose_65/BiasAdd/ReadVariableOp2�
Asequential_28/conv2d_transpose_65/conv2d_transpose/ReadVariableOpAsequential_28/conv2d_transpose_65/conv2d_transpose/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:Y U
/
_output_shapes
:���������
"
_user_specified_name
input_30
�"
�
P__inference_conv2d_transpose_64_layer_call_and_return_conditional_losses_3492203

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+��������������������������� {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� ]
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
+__inference_conv2d_42_layer_call_fn_3492497

inputs!
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_42_layer_call_and_return_conditional_losses_3492267w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	3492493:'#
!
_user_specified_name	3492491:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
%__inference_signature_wrapper_3492488
input_30!
unknown: 
	unknown_0: #
	unknown_1: 
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: #
	unknown_7: 
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_30unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_3492126w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'
#
!
_user_specified_name	3492484:'	#
!
_user_specified_name	3492482:'#
!
_user_specified_name	3492480:'#
!
_user_specified_name	3492478:'#
!
_user_specified_name	3492476:'#
!
_user_specified_name	3492474:'#
!
_user_specified_name	3492472:'#
!
_user_specified_name	3492470:'#
!
_user_specified_name	3492468:'#
!
_user_specified_name	3492466:Y U
/
_output_shapes
:���������
"
_user_specified_name
input_30
�
H
,__inference_dropout_59_layer_call_fn_3492608

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_59_layer_call_and_return_conditional_losses_3492359h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
e
G__inference_dropout_59_layer_call_and_return_conditional_losses_3492359

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_conv2d_42_layer_call_and_return_conditional_losses_3492508

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:��������� i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:��������� S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�"
�
P__inference_conv2d_transpose_63_layer_call_and_return_conditional_losses_3492598

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������]
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
e
G__inference_dropout_59_layer_call_and_return_conditional_losses_3492625

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
e
,__inference_dropout_59_layer_call_fn_3492603

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_59_layer_call_and_return_conditional_losses_3492318w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�

f
G__inference_dropout_59_layer_call_and_return_conditional_losses_3492318

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�

f
G__inference_dropout_58_layer_call_and_return_conditional_losses_3492284

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:��������� Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:��������� *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:��������� T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:��������� i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
F__inference_conv2d_43_layer_call_and_return_conditional_losses_3492555

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

f
G__inference_dropout_59_layer_call_and_return_conditional_losses_3492620

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�!
�
P__inference_conv2d_transpose_65_layer_call_and_return_conditional_losses_3492245

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������]
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�"
�
P__inference_conv2d_transpose_64_layer_call_and_return_conditional_losses_3492668

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+��������������������������� {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� ]
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
F__inference_conv2d_43_layer_call_and_return_conditional_losses_3492296

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�"
�
P__inference_conv2d_transpose_63_layer_call_and_return_conditional_losses_3492160

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������]
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
5__inference_conv2d_transpose_64_layer_call_fn_3492634

inputs!
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_conv2d_transpose_64_layer_call_and_return_conditional_losses_3492203�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	3492630:'#
!
_user_specified_name	3492628:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
e
,__inference_dropout_58_layer_call_fn_3492513

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_58_layer_call_and_return_conditional_losses_3492284w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
H
,__inference_dropout_58_layer_call_fn_3492518

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_58_layer_call_and_return_conditional_losses_3492343h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�(
�
J__inference_sequential_28_layer_call_and_return_conditional_losses_3492372
input_30+
conv2d_42_3492334: 
conv2d_42_3492336: +
conv2d_43_3492345: 
conv2d_43_3492347:5
conv2d_transpose_63_3492350:)
conv2d_transpose_63_3492352:5
conv2d_transpose_64_3492361: )
conv2d_transpose_64_3492363: 5
conv2d_transpose_65_3492366: )
conv2d_transpose_65_3492368:
identity��!conv2d_42/StatefulPartitionedCall�!conv2d_43/StatefulPartitionedCall�+conv2d_transpose_63/StatefulPartitionedCall�+conv2d_transpose_64/StatefulPartitionedCall�+conv2d_transpose_65/StatefulPartitionedCall�
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCallinput_30conv2d_42_3492334conv2d_42_3492336*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_42_layer_call_and_return_conditional_losses_3492267�
dropout_58/PartitionedCallPartitionedCall*conv2d_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_58_layer_call_and_return_conditional_losses_3492343�
!conv2d_43/StatefulPartitionedCallStatefulPartitionedCall#dropout_58/PartitionedCall:output:0conv2d_43_3492345conv2d_43_3492347*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_43_layer_call_and_return_conditional_losses_3492296�
+conv2d_transpose_63/StatefulPartitionedCallStatefulPartitionedCall*conv2d_43/StatefulPartitionedCall:output:0conv2d_transpose_63_3492350conv2d_transpose_63_3492352*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_conv2d_transpose_63_layer_call_and_return_conditional_losses_3492160�
dropout_59/PartitionedCallPartitionedCall4conv2d_transpose_63/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_59_layer_call_and_return_conditional_losses_3492359�
+conv2d_transpose_64/StatefulPartitionedCallStatefulPartitionedCall#dropout_59/PartitionedCall:output:0conv2d_transpose_64_3492361conv2d_transpose_64_3492363*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_conv2d_transpose_64_layer_call_and_return_conditional_losses_3492203�
+conv2d_transpose_65/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_64/StatefulPartitionedCall:output:0conv2d_transpose_65_3492366conv2d_transpose_65_3492368*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_conv2d_transpose_65_layer_call_and_return_conditional_losses_3492245�
IdentityIdentity4conv2d_transpose_65/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp"^conv2d_42/StatefulPartitionedCall"^conv2d_43/StatefulPartitionedCall,^conv2d_transpose_63/StatefulPartitionedCall,^conv2d_transpose_64/StatefulPartitionedCall,^conv2d_transpose_65/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : 2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall2F
!conv2d_43/StatefulPartitionedCall!conv2d_43/StatefulPartitionedCall2Z
+conv2d_transpose_63/StatefulPartitionedCall+conv2d_transpose_63/StatefulPartitionedCall2Z
+conv2d_transpose_64/StatefulPartitionedCall+conv2d_transpose_64/StatefulPartitionedCall2Z
+conv2d_transpose_65/StatefulPartitionedCall+conv2d_transpose_65/StatefulPartitionedCall:'
#
!
_user_specified_name	3492368:'	#
!
_user_specified_name	3492366:'#
!
_user_specified_name	3492363:'#
!
_user_specified_name	3492361:'#
!
_user_specified_name	3492352:'#
!
_user_specified_name	3492350:'#
!
_user_specified_name	3492347:'#
!
_user_specified_name	3492345:'#
!
_user_specified_name	3492336:'#
!
_user_specified_name	3492334:Y U
/
_output_shapes
:���������
"
_user_specified_name
input_30
�
�
5__inference_conv2d_transpose_63_layer_call_fn_3492564

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_conv2d_transpose_63_layer_call_and_return_conditional_losses_3492160�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	3492560:'#
!
_user_specified_name	3492558:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
/__inference_sequential_28_layer_call_fn_3492422
input_30!
unknown: 
	unknown_0: #
	unknown_1: 
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: #
	unknown_7: 
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_30unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_28_layer_call_and_return_conditional_losses_3492372w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'
#
!
_user_specified_name	3492418:'	#
!
_user_specified_name	3492416:'#
!
_user_specified_name	3492414:'#
!
_user_specified_name	3492412:'#
!
_user_specified_name	3492410:'#
!
_user_specified_name	3492408:'#
!
_user_specified_name	3492406:'#
!
_user_specified_name	3492404:'#
!
_user_specified_name	3492402:'#
!
_user_specified_name	3492400:Y U
/
_output_shapes
:���������
"
_user_specified_name
input_30
ã
�
#__inference__traced_restore_3493047
file_prefix;
!assignvariableop_conv2d_42_kernel: /
!assignvariableop_1_conv2d_42_bias: =
#assignvariableop_2_conv2d_43_kernel: /
!assignvariableop_3_conv2d_43_bias:G
-assignvariableop_4_conv2d_transpose_63_kernel:9
+assignvariableop_5_conv2d_transpose_63_bias:G
-assignvariableop_6_conv2d_transpose_64_kernel: 9
+assignvariableop_7_conv2d_transpose_64_bias: G
-assignvariableop_8_conv2d_transpose_65_kernel: 9
+assignvariableop_9_conv2d_transpose_65_bias:'
assignvariableop_10_iteration:	 +
!assignvariableop_11_learning_rate: E
+assignvariableop_12_adam_m_conv2d_42_kernel: E
+assignvariableop_13_adam_v_conv2d_42_kernel: 7
)assignvariableop_14_adam_m_conv2d_42_bias: 7
)assignvariableop_15_adam_v_conv2d_42_bias: E
+assignvariableop_16_adam_m_conv2d_43_kernel: E
+assignvariableop_17_adam_v_conv2d_43_kernel: 7
)assignvariableop_18_adam_m_conv2d_43_bias:7
)assignvariableop_19_adam_v_conv2d_43_bias:O
5assignvariableop_20_adam_m_conv2d_transpose_63_kernel:O
5assignvariableop_21_adam_v_conv2d_transpose_63_kernel:A
3assignvariableop_22_adam_m_conv2d_transpose_63_bias:A
3assignvariableop_23_adam_v_conv2d_transpose_63_bias:O
5assignvariableop_24_adam_m_conv2d_transpose_64_kernel: O
5assignvariableop_25_adam_v_conv2d_transpose_64_kernel: A
3assignvariableop_26_adam_m_conv2d_transpose_64_bias: A
3assignvariableop_27_adam_v_conv2d_transpose_64_bias: O
5assignvariableop_28_adam_m_conv2d_transpose_65_kernel: O
5assignvariableop_29_adam_v_conv2d_transpose_65_kernel: A
3assignvariableop_30_adam_m_conv2d_transpose_65_bias:A
3assignvariableop_31_adam_v_conv2d_transpose_65_bias:#
assignvariableop_32_total: #
assignvariableop_33_count: 
identity_35��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*�
value�B�#B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::*1
dtypes'
%2#	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_42_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_42_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_43_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_43_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp-assignvariableop_4_conv2d_transpose_63_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp+assignvariableop_5_conv2d_transpose_63_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp-assignvariableop_6_conv2d_transpose_64_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp+assignvariableop_7_conv2d_transpose_64_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp-assignvariableop_8_conv2d_transpose_65_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp+assignvariableop_9_conv2d_transpose_65_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_iterationIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp!assignvariableop_11_learning_rateIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp+assignvariableop_12_adam_m_conv2d_42_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp+assignvariableop_13_adam_v_conv2d_42_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp)assignvariableop_14_adam_m_conv2d_42_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp)assignvariableop_15_adam_v_conv2d_42_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp+assignvariableop_16_adam_m_conv2d_43_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp+assignvariableop_17_adam_v_conv2d_43_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp)assignvariableop_18_adam_m_conv2d_43_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_v_conv2d_43_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp5assignvariableop_20_adam_m_conv2d_transpose_63_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp5assignvariableop_21_adam_v_conv2d_transpose_63_kernelIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp3assignvariableop_22_adam_m_conv2d_transpose_63_biasIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp3assignvariableop_23_adam_v_conv2d_transpose_63_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp5assignvariableop_24_adam_m_conv2d_transpose_64_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp5assignvariableop_25_adam_v_conv2d_transpose_64_kernelIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp3assignvariableop_26_adam_m_conv2d_transpose_64_biasIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp3assignvariableop_27_adam_v_conv2d_transpose_64_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp5assignvariableop_28_adam_m_conv2d_transpose_65_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp5assignvariableop_29_adam_v_conv2d_transpose_65_kernelIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp3assignvariableop_30_adam_m_conv2d_transpose_65_biasIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp3assignvariableop_31_adam_v_conv2d_transpose_65_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOpassignvariableop_32_totalIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOpassignvariableop_33_countIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_34Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_35IdentityIdentity_34:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_35Identity_35:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_1AssignVariableOp_12*
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
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%"!

_user_specified_namecount:%!!

_user_specified_nametotal:? ;
9
_user_specified_name!Adam/v/conv2d_transpose_65/bias:?;
9
_user_specified_name!Adam/m/conv2d_transpose_65/bias:A=
;
_user_specified_name#!Adam/v/conv2d_transpose_65/kernel:A=
;
_user_specified_name#!Adam/m/conv2d_transpose_65/kernel:?;
9
_user_specified_name!Adam/v/conv2d_transpose_64/bias:?;
9
_user_specified_name!Adam/m/conv2d_transpose_64/bias:A=
;
_user_specified_name#!Adam/v/conv2d_transpose_64/kernel:A=
;
_user_specified_name#!Adam/m/conv2d_transpose_64/kernel:?;
9
_user_specified_name!Adam/v/conv2d_transpose_63/bias:?;
9
_user_specified_name!Adam/m/conv2d_transpose_63/bias:A=
;
_user_specified_name#!Adam/v/conv2d_transpose_63/kernel:A=
;
_user_specified_name#!Adam/m/conv2d_transpose_63/kernel:51
/
_user_specified_nameAdam/v/conv2d_43/bias:51
/
_user_specified_nameAdam/m/conv2d_43/bias:73
1
_user_specified_nameAdam/v/conv2d_43/kernel:73
1
_user_specified_nameAdam/m/conv2d_43/kernel:51
/
_user_specified_nameAdam/v/conv2d_42/bias:51
/
_user_specified_nameAdam/m/conv2d_42/bias:73
1
_user_specified_nameAdam/v/conv2d_42/kernel:73
1
_user_specified_nameAdam/m/conv2d_42/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:8
4
2
_user_specified_nameconv2d_transpose_65/bias::	6
4
_user_specified_nameconv2d_transpose_65/kernel:84
2
_user_specified_nameconv2d_transpose_64/bias::6
4
_user_specified_nameconv2d_transpose_64/kernel:84
2
_user_specified_nameconv2d_transpose_63/bias::6
4
_user_specified_nameconv2d_transpose_63/kernel:.*
(
_user_specified_nameconv2d_43/bias:0,
*
_user_specified_nameconv2d_43/kernel:.*
(
_user_specified_nameconv2d_42/bias:0,
*
_user_specified_nameconv2d_42/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
+__inference_conv2d_43_layer_call_fn_3492544

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_43_layer_call_and_return_conditional_losses_3492296w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	3492540:'#
!
_user_specified_name	3492538:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

f
G__inference_dropout_58_layer_call_and_return_conditional_losses_3492530

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:��������� Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:��������� *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:��������� T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:��������� i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
F__inference_conv2d_42_layer_call_and_return_conditional_losses_3492267

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:��������� i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:��������� S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
e
G__inference_dropout_58_layer_call_and_return_conditional_losses_3492535

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:��������� c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
5__inference_conv2d_transpose_65_layer_call_fn_3492677

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_conv2d_transpose_65_layer_call_and_return_conditional_losses_3492245�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	3492673:'#
!
_user_specified_name	3492671:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�+
�
J__inference_sequential_28_layer_call_and_return_conditional_losses_3492331
input_30+
conv2d_42_3492268: 
conv2d_42_3492270: +
conv2d_43_3492297: 
conv2d_43_3492299:5
conv2d_transpose_63_3492302:)
conv2d_transpose_63_3492304:5
conv2d_transpose_64_3492320: )
conv2d_transpose_64_3492322: 5
conv2d_transpose_65_3492325: )
conv2d_transpose_65_3492327:
identity��!conv2d_42/StatefulPartitionedCall�!conv2d_43/StatefulPartitionedCall�+conv2d_transpose_63/StatefulPartitionedCall�+conv2d_transpose_64/StatefulPartitionedCall�+conv2d_transpose_65/StatefulPartitionedCall�"dropout_58/StatefulPartitionedCall�"dropout_59/StatefulPartitionedCall�
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCallinput_30conv2d_42_3492268conv2d_42_3492270*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_42_layer_call_and_return_conditional_losses_3492267�
"dropout_58/StatefulPartitionedCallStatefulPartitionedCall*conv2d_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_58_layer_call_and_return_conditional_losses_3492284�
!conv2d_43/StatefulPartitionedCallStatefulPartitionedCall+dropout_58/StatefulPartitionedCall:output:0conv2d_43_3492297conv2d_43_3492299*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_43_layer_call_and_return_conditional_losses_3492296�
+conv2d_transpose_63/StatefulPartitionedCallStatefulPartitionedCall*conv2d_43/StatefulPartitionedCall:output:0conv2d_transpose_63_3492302conv2d_transpose_63_3492304*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_conv2d_transpose_63_layer_call_and_return_conditional_losses_3492160�
"dropout_59/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_63/StatefulPartitionedCall:output:0#^dropout_58/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_59_layer_call_and_return_conditional_losses_3492318�
+conv2d_transpose_64/StatefulPartitionedCallStatefulPartitionedCall+dropout_59/StatefulPartitionedCall:output:0conv2d_transpose_64_3492320conv2d_transpose_64_3492322*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_conv2d_transpose_64_layer_call_and_return_conditional_losses_3492203�
+conv2d_transpose_65/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_64/StatefulPartitionedCall:output:0conv2d_transpose_65_3492325conv2d_transpose_65_3492327*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_conv2d_transpose_65_layer_call_and_return_conditional_losses_3492245�
IdentityIdentity4conv2d_transpose_65/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp"^conv2d_42/StatefulPartitionedCall"^conv2d_43/StatefulPartitionedCall,^conv2d_transpose_63/StatefulPartitionedCall,^conv2d_transpose_64/StatefulPartitionedCall,^conv2d_transpose_65/StatefulPartitionedCall#^dropout_58/StatefulPartitionedCall#^dropout_59/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : 2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall2F
!conv2d_43/StatefulPartitionedCall!conv2d_43/StatefulPartitionedCall2Z
+conv2d_transpose_63/StatefulPartitionedCall+conv2d_transpose_63/StatefulPartitionedCall2Z
+conv2d_transpose_64/StatefulPartitionedCall+conv2d_transpose_64/StatefulPartitionedCall2Z
+conv2d_transpose_65/StatefulPartitionedCall+conv2d_transpose_65/StatefulPartitionedCall2H
"dropout_58/StatefulPartitionedCall"dropout_58/StatefulPartitionedCall2H
"dropout_59/StatefulPartitionedCall"dropout_59/StatefulPartitionedCall:'
#
!
_user_specified_name	3492327:'	#
!
_user_specified_name	3492325:'#
!
_user_specified_name	3492322:'#
!
_user_specified_name	3492320:'#
!
_user_specified_name	3492304:'#
!
_user_specified_name	3492302:'#
!
_user_specified_name	3492299:'#
!
_user_specified_name	3492297:'#
!
_user_specified_name	3492270:'#
!
_user_specified_name	3492268:Y U
/
_output_shapes
:���������
"
_user_specified_name
input_30
��
�!
 __inference__traced_save_3492936
file_prefixA
'read_disablecopyonread_conv2d_42_kernel: 5
'read_1_disablecopyonread_conv2d_42_bias: C
)read_2_disablecopyonread_conv2d_43_kernel: 5
'read_3_disablecopyonread_conv2d_43_bias:M
3read_4_disablecopyonread_conv2d_transpose_63_kernel:?
1read_5_disablecopyonread_conv2d_transpose_63_bias:M
3read_6_disablecopyonread_conv2d_transpose_64_kernel: ?
1read_7_disablecopyonread_conv2d_transpose_64_bias: M
3read_8_disablecopyonread_conv2d_transpose_65_kernel: ?
1read_9_disablecopyonread_conv2d_transpose_65_bias:-
#read_10_disablecopyonread_iteration:	 1
'read_11_disablecopyonread_learning_rate: K
1read_12_disablecopyonread_adam_m_conv2d_42_kernel: K
1read_13_disablecopyonread_adam_v_conv2d_42_kernel: =
/read_14_disablecopyonread_adam_m_conv2d_42_bias: =
/read_15_disablecopyonread_adam_v_conv2d_42_bias: K
1read_16_disablecopyonread_adam_m_conv2d_43_kernel: K
1read_17_disablecopyonread_adam_v_conv2d_43_kernel: =
/read_18_disablecopyonread_adam_m_conv2d_43_bias:=
/read_19_disablecopyonread_adam_v_conv2d_43_bias:U
;read_20_disablecopyonread_adam_m_conv2d_transpose_63_kernel:U
;read_21_disablecopyonread_adam_v_conv2d_transpose_63_kernel:G
9read_22_disablecopyonread_adam_m_conv2d_transpose_63_bias:G
9read_23_disablecopyonread_adam_v_conv2d_transpose_63_bias:U
;read_24_disablecopyonread_adam_m_conv2d_transpose_64_kernel: U
;read_25_disablecopyonread_adam_v_conv2d_transpose_64_kernel: G
9read_26_disablecopyonread_adam_m_conv2d_transpose_64_bias: G
9read_27_disablecopyonread_adam_v_conv2d_transpose_64_bias: U
;read_28_disablecopyonread_adam_m_conv2d_transpose_65_kernel: U
;read_29_disablecopyonread_adam_v_conv2d_transpose_65_kernel: G
9read_30_disablecopyonread_adam_m_conv2d_transpose_65_bias:G
9read_31_disablecopyonread_adam_v_conv2d_transpose_65_bias:)
read_32_disablecopyonread_total: )
read_33_disablecopyonread_count: 
savev2_const
identity_69��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: y
Read/DisableCopyOnReadDisableCopyOnRead'read_disablecopyonread_conv2d_42_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp'read_disablecopyonread_conv2d_42_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
: {
Read_1/DisableCopyOnReadDisableCopyOnRead'read_1_disablecopyonread_conv2d_42_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp'read_1_disablecopyonread_conv2d_42_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
: }
Read_2/DisableCopyOnReadDisableCopyOnRead)read_2_disablecopyonread_conv2d_43_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp)read_2_disablecopyonread_conv2d_43_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0u

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: k

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*&
_output_shapes
: {
Read_3/DisableCopyOnReadDisableCopyOnRead'read_3_disablecopyonread_conv2d_43_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp'read_3_disablecopyonread_conv2d_43_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_4/DisableCopyOnReadDisableCopyOnRead3read_4_disablecopyonread_conv2d_transpose_63_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp3read_4_disablecopyonread_conv2d_transpose_63_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0u

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:k

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_5/DisableCopyOnReadDisableCopyOnRead1read_5_disablecopyonread_conv2d_transpose_63_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp1read_5_disablecopyonread_conv2d_transpose_63_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_6/DisableCopyOnReadDisableCopyOnRead3read_6_disablecopyonread_conv2d_transpose_64_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp3read_6_disablecopyonread_conv2d_transpose_64_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0v
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_7/DisableCopyOnReadDisableCopyOnRead1read_7_disablecopyonread_conv2d_transpose_64_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp1read_7_disablecopyonread_conv2d_transpose_64_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_8/DisableCopyOnReadDisableCopyOnRead3read_8_disablecopyonread_conv2d_transpose_65_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp3read_8_disablecopyonread_conv2d_transpose_65_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0v
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_9/DisableCopyOnReadDisableCopyOnRead1read_9_disablecopyonread_conv2d_transpose_65_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp1read_9_disablecopyonread_conv2d_transpose_65_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_10/DisableCopyOnReadDisableCopyOnRead#read_10_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp#read_10_disablecopyonread_iteration^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_11/DisableCopyOnReadDisableCopyOnRead'read_11_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp'read_11_disablecopyonread_learning_rate^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_12/DisableCopyOnReadDisableCopyOnRead1read_12_disablecopyonread_adam_m_conv2d_42_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp1read_12_disablecopyonread_adam_m_conv2d_42_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_13/DisableCopyOnReadDisableCopyOnRead1read_13_disablecopyonread_adam_v_conv2d_42_kernel"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp1read_13_disablecopyonread_adam_v_conv2d_42_kernel^Read_13/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_14/DisableCopyOnReadDisableCopyOnRead/read_14_disablecopyonread_adam_m_conv2d_42_bias"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp/read_14_disablecopyonread_adam_m_conv2d_42_bias^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_15/DisableCopyOnReadDisableCopyOnRead/read_15_disablecopyonread_adam_v_conv2d_42_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp/read_15_disablecopyonread_adam_v_conv2d_42_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_16/DisableCopyOnReadDisableCopyOnRead1read_16_disablecopyonread_adam_m_conv2d_43_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp1read_16_disablecopyonread_adam_m_conv2d_43_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_17/DisableCopyOnReadDisableCopyOnRead1read_17_disablecopyonread_adam_v_conv2d_43_kernel"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp1read_17_disablecopyonread_adam_v_conv2d_43_kernel^Read_17/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_18/DisableCopyOnReadDisableCopyOnRead/read_18_disablecopyonread_adam_m_conv2d_43_bias"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp/read_18_disablecopyonread_adam_m_conv2d_43_bias^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_19/DisableCopyOnReadDisableCopyOnRead/read_19_disablecopyonread_adam_v_conv2d_43_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp/read_19_disablecopyonread_adam_v_conv2d_43_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_20/DisableCopyOnReadDisableCopyOnRead;read_20_disablecopyonread_adam_m_conv2d_transpose_63_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp;read_20_disablecopyonread_adam_m_conv2d_transpose_63_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_21/DisableCopyOnReadDisableCopyOnRead;read_21_disablecopyonread_adam_v_conv2d_transpose_63_kernel"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp;read_21_disablecopyonread_adam_v_conv2d_transpose_63_kernel^Read_21/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_22/DisableCopyOnReadDisableCopyOnRead9read_22_disablecopyonread_adam_m_conv2d_transpose_63_bias"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp9read_22_disablecopyonread_adam_m_conv2d_transpose_63_bias^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_23/DisableCopyOnReadDisableCopyOnRead9read_23_disablecopyonread_adam_v_conv2d_transpose_63_bias"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp9read_23_disablecopyonread_adam_v_conv2d_transpose_63_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_24/DisableCopyOnReadDisableCopyOnRead;read_24_disablecopyonread_adam_m_conv2d_transpose_64_kernel"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp;read_24_disablecopyonread_adam_m_conv2d_transpose_64_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_25/DisableCopyOnReadDisableCopyOnRead;read_25_disablecopyonread_adam_v_conv2d_transpose_64_kernel"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp;read_25_disablecopyonread_adam_v_conv2d_transpose_64_kernel^Read_25/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_26/DisableCopyOnReadDisableCopyOnRead9read_26_disablecopyonread_adam_m_conv2d_transpose_64_bias"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp9read_26_disablecopyonread_adam_m_conv2d_transpose_64_bias^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_27/DisableCopyOnReadDisableCopyOnRead9read_27_disablecopyonread_adam_v_conv2d_transpose_64_bias"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp9read_27_disablecopyonread_adam_v_conv2d_transpose_64_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_28/DisableCopyOnReadDisableCopyOnRead;read_28_disablecopyonread_adam_m_conv2d_transpose_65_kernel"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp;read_28_disablecopyonread_adam_m_conv2d_transpose_65_kernel^Read_28/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_29/DisableCopyOnReadDisableCopyOnRead;read_29_disablecopyonread_adam_v_conv2d_transpose_65_kernel"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp;read_29_disablecopyonread_adam_v_conv2d_transpose_65_kernel^Read_29/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_30/DisableCopyOnReadDisableCopyOnRead9read_30_disablecopyonread_adam_m_conv2d_transpose_65_bias"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp9read_30_disablecopyonread_adam_m_conv2d_transpose_65_bias^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_31/DisableCopyOnReadDisableCopyOnRead9read_31_disablecopyonread_adam_v_conv2d_transpose_65_bias"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp9read_31_disablecopyonread_adam_v_conv2d_transpose_65_bias^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:t
Read_32/DisableCopyOnReadDisableCopyOnReadread_32_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOpread_32_disablecopyonread_total^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_33/DisableCopyOnReadDisableCopyOnReadread_33_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOpread_33_disablecopyonread_count^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*�
value�B�#B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *1
dtypes'
%2#	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_68Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_69IdentityIdentity_68:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_69Identity_69:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=#9

_output_shapes
: 

_user_specified_nameConst:%"!

_user_specified_namecount:%!!

_user_specified_nametotal:? ;
9
_user_specified_name!Adam/v/conv2d_transpose_65/bias:?;
9
_user_specified_name!Adam/m/conv2d_transpose_65/bias:A=
;
_user_specified_name#!Adam/v/conv2d_transpose_65/kernel:A=
;
_user_specified_name#!Adam/m/conv2d_transpose_65/kernel:?;
9
_user_specified_name!Adam/v/conv2d_transpose_64/bias:?;
9
_user_specified_name!Adam/m/conv2d_transpose_64/bias:A=
;
_user_specified_name#!Adam/v/conv2d_transpose_64/kernel:A=
;
_user_specified_name#!Adam/m/conv2d_transpose_64/kernel:?;
9
_user_specified_name!Adam/v/conv2d_transpose_63/bias:?;
9
_user_specified_name!Adam/m/conv2d_transpose_63/bias:A=
;
_user_specified_name#!Adam/v/conv2d_transpose_63/kernel:A=
;
_user_specified_name#!Adam/m/conv2d_transpose_63/kernel:51
/
_user_specified_nameAdam/v/conv2d_43/bias:51
/
_user_specified_nameAdam/m/conv2d_43/bias:73
1
_user_specified_nameAdam/v/conv2d_43/kernel:73
1
_user_specified_nameAdam/m/conv2d_43/kernel:51
/
_user_specified_nameAdam/v/conv2d_42/bias:51
/
_user_specified_nameAdam/m/conv2d_42/bias:73
1
_user_specified_nameAdam/v/conv2d_42/kernel:73
1
_user_specified_nameAdam/m/conv2d_42/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:8
4
2
_user_specified_nameconv2d_transpose_65/bias::	6
4
_user_specified_nameconv2d_transpose_65/kernel:84
2
_user_specified_nameconv2d_transpose_64/bias::6
4
_user_specified_nameconv2d_transpose_64/kernel:84
2
_user_specified_nameconv2d_transpose_63/bias::6
4
_user_specified_nameconv2d_transpose_63/kernel:.*
(
_user_specified_nameconv2d_43/bias:0,
*
_user_specified_nameconv2d_43/kernel:.*
(
_user_specified_nameconv2d_42/bias:0,
*
_user_specified_nameconv2d_42/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
E
input_309
serving_default_input_30:0���������O
conv2d_transpose_658
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
 _random_generator"
_tf_keras_layer
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses

'kernel
(bias
 )_jit_compiled_convolution_op"
_tf_keras_layer
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses

0kernel
1bias
 2_jit_compiled_convolution_op"
_tf_keras_layer
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses
9_random_generator"
_tf_keras_layer
�
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

@kernel
Abias
 B_jit_compiled_convolution_op"
_tf_keras_layer
�
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses

Ikernel
Jbias
 K_jit_compiled_convolution_op"
_tf_keras_layer
f
0
1
'2
(3
04
15
@6
A7
I8
J9"
trackable_list_wrapper
f
0
1
'2
(3
04
15
@6
A7
I8
J9"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Qtrace_0
Rtrace_12�
/__inference_sequential_28_layer_call_fn_3492397
/__inference_sequential_28_layer_call_fn_3492422�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zQtrace_0zRtrace_1
�
Strace_0
Ttrace_12�
J__inference_sequential_28_layer_call_and_return_conditional_losses_3492331
J__inference_sequential_28_layer_call_and_return_conditional_losses_3492372�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zStrace_0zTtrace_1
�B�
"__inference__wrapped_model_3492126input_30"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
U
_variables
V_iterations
W_learning_rate
X_index_dict
Y
_momentums
Z_velocities
[_update_step_xla"
experimentalOptimizer
,
\serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
btrace_02�
+__inference_conv2d_42_layer_call_fn_3492497�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zbtrace_0
�
ctrace_02�
F__inference_conv2d_42_layer_call_and_return_conditional_losses_3492508�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zctrace_0
*:( 2conv2d_42/kernel
: 2conv2d_42/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
itrace_0
jtrace_12�
,__inference_dropout_58_layer_call_fn_3492513
,__inference_dropout_58_layer_call_fn_3492518�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zitrace_0zjtrace_1
�
ktrace_0
ltrace_12�
G__inference_dropout_58_layer_call_and_return_conditional_losses_3492530
G__inference_dropout_58_layer_call_and_return_conditional_losses_3492535�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zktrace_0zltrace_1
"
_generic_user_object
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
�
rtrace_02�
+__inference_conv2d_43_layer_call_fn_3492544�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zrtrace_0
�
strace_02�
F__inference_conv2d_43_layer_call_and_return_conditional_losses_3492555�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zstrace_0
*:( 2conv2d_43/kernel
:2conv2d_43/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
�
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
�
ytrace_02�
5__inference_conv2d_transpose_63_layer_call_fn_3492564�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zytrace_0
�
ztrace_02�
P__inference_conv2d_transpose_63_layer_call_and_return_conditional_losses_3492598�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zztrace_0
4:22conv2d_transpose_63/kernel
&:$2conv2d_transpose_63/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
,__inference_dropout_59_layer_call_fn_3492603
,__inference_dropout_59_layer_call_fn_3492608�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
G__inference_dropout_59_layer_call_and_return_conditional_losses_3492620
G__inference_dropout_59_layer_call_and_return_conditional_losses_3492625�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
5__inference_conv2d_transpose_64_layer_call_fn_3492634�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
P__inference_conv2d_transpose_64_layer_call_and_return_conditional_losses_3492668�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
4:2 2conv2d_transpose_64/kernel
&:$ 2conv2d_transpose_64/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
5__inference_conv2d_transpose_65_layer_call_fn_3492677�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
P__inference_conv2d_transpose_65_layer_call_and_return_conditional_losses_3492710�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
4:2 2conv2d_transpose_65/kernel
&:$2conv2d_transpose_65/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
/__inference_sequential_28_layer_call_fn_3492397input_30"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
/__inference_sequential_28_layer_call_fn_3492422input_30"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_sequential_28_layer_call_and_return_conditional_losses_3492331input_30"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_sequential_28_layer_call_and_return_conditional_losses_3492372input_30"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
V0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
p
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9"
trackable_list_wrapper
p
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
%__inference_signature_wrapper_3492488input_30"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
+__inference_conv2d_42_layer_call_fn_3492497inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_conv2d_42_layer_call_and_return_conditional_losses_3492508inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
,__inference_dropout_58_layer_call_fn_3492513inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_dropout_58_layer_call_fn_3492518inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dropout_58_layer_call_and_return_conditional_losses_3492530inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dropout_58_layer_call_and_return_conditional_losses_3492535inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
+__inference_conv2d_43_layer_call_fn_3492544inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_conv2d_43_layer_call_and_return_conditional_losses_3492555inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
5__inference_conv2d_transpose_63_layer_call_fn_3492564inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
P__inference_conv2d_transpose_63_layer_call_and_return_conditional_losses_3492598inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
,__inference_dropout_59_layer_call_fn_3492603inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_dropout_59_layer_call_fn_3492608inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dropout_59_layer_call_and_return_conditional_losses_3492620inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dropout_59_layer_call_and_return_conditional_losses_3492625inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
5__inference_conv2d_transpose_64_layer_call_fn_3492634inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
P__inference_conv2d_transpose_64_layer_call_and_return_conditional_losses_3492668inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
5__inference_conv2d_transpose_65_layer_call_fn_3492677inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
P__inference_conv2d_transpose_65_layer_call_and_return_conditional_losses_3492710inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
/:- 2Adam/m/conv2d_42/kernel
/:- 2Adam/v/conv2d_42/kernel
!: 2Adam/m/conv2d_42/bias
!: 2Adam/v/conv2d_42/bias
/:- 2Adam/m/conv2d_43/kernel
/:- 2Adam/v/conv2d_43/kernel
!:2Adam/m/conv2d_43/bias
!:2Adam/v/conv2d_43/bias
9:72!Adam/m/conv2d_transpose_63/kernel
9:72!Adam/v/conv2d_transpose_63/kernel
+:)2Adam/m/conv2d_transpose_63/bias
+:)2Adam/v/conv2d_transpose_63/bias
9:7 2!Adam/m/conv2d_transpose_64/kernel
9:7 2!Adam/v/conv2d_transpose_64/kernel
+:) 2Adam/m/conv2d_transpose_64/bias
+:) 2Adam/v/conv2d_transpose_64/bias
9:7 2!Adam/m/conv2d_transpose_65/kernel
9:7 2!Adam/v/conv2d_transpose_65/kernel
+:)2Adam/m/conv2d_transpose_65/bias
+:)2Adam/v/conv2d_transpose_65/bias
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count�
"__inference__wrapped_model_3492126�
'(01@AIJ9�6
/�,
*�'
input_30���������
� "Q�N
L
conv2d_transpose_655�2
conv2d_transpose_65����������
F__inference_conv2d_42_layer_call_and_return_conditional_losses_3492508s7�4
-�*
(�%
inputs���������
� "4�1
*�'
tensor_0��������� 
� �
+__inference_conv2d_42_layer_call_fn_3492497h7�4
-�*
(�%
inputs���������
� ")�&
unknown��������� �
F__inference_conv2d_43_layer_call_and_return_conditional_losses_3492555s'(7�4
-�*
(�%
inputs��������� 
� "4�1
*�'
tensor_0���������
� �
+__inference_conv2d_43_layer_call_fn_3492544h'(7�4
-�*
(�%
inputs��������� 
� ")�&
unknown����������
P__inference_conv2d_transpose_63_layer_call_and_return_conditional_losses_3492598�01I�F
?�<
:�7
inputs+���������������������������
� "F�C
<�9
tensor_0+���������������������������
� �
5__inference_conv2d_transpose_63_layer_call_fn_3492564�01I�F
?�<
:�7
inputs+���������������������������
� ";�8
unknown+����������������������������
P__inference_conv2d_transpose_64_layer_call_and_return_conditional_losses_3492668�@AI�F
?�<
:�7
inputs+���������������������������
� "F�C
<�9
tensor_0+��������������������������� 
� �
5__inference_conv2d_transpose_64_layer_call_fn_3492634�@AI�F
?�<
:�7
inputs+���������������������������
� ";�8
unknown+��������������������������� �
P__inference_conv2d_transpose_65_layer_call_and_return_conditional_losses_3492710�IJI�F
?�<
:�7
inputs+��������������������������� 
� "F�C
<�9
tensor_0+���������������������������
� �
5__inference_conv2d_transpose_65_layer_call_fn_3492677�IJI�F
?�<
:�7
inputs+��������������������������� 
� ";�8
unknown+����������������������������
G__inference_dropout_58_layer_call_and_return_conditional_losses_3492530s;�8
1�.
(�%
inputs��������� 
p
� "4�1
*�'
tensor_0��������� 
� �
G__inference_dropout_58_layer_call_and_return_conditional_losses_3492535s;�8
1�.
(�%
inputs��������� 
p 
� "4�1
*�'
tensor_0��������� 
� �
,__inference_dropout_58_layer_call_fn_3492513h;�8
1�.
(�%
inputs��������� 
p
� ")�&
unknown��������� �
,__inference_dropout_58_layer_call_fn_3492518h;�8
1�.
(�%
inputs��������� 
p 
� ")�&
unknown��������� �
G__inference_dropout_59_layer_call_and_return_conditional_losses_3492620s;�8
1�.
(�%
inputs���������
p
� "4�1
*�'
tensor_0���������
� �
G__inference_dropout_59_layer_call_and_return_conditional_losses_3492625s;�8
1�.
(�%
inputs���������
p 
� "4�1
*�'
tensor_0���������
� �
,__inference_dropout_59_layer_call_fn_3492603h;�8
1�.
(�%
inputs���������
p
� ")�&
unknown����������
,__inference_dropout_59_layer_call_fn_3492608h;�8
1�.
(�%
inputs���������
p 
� ")�&
unknown����������
J__inference_sequential_28_layer_call_and_return_conditional_losses_3492331�
'(01@AIJA�>
7�4
*�'
input_30���������
p

 
� "4�1
*�'
tensor_0���������
� �
J__inference_sequential_28_layer_call_and_return_conditional_losses_3492372�
'(01@AIJA�>
7�4
*�'
input_30���������
p 

 
� "4�1
*�'
tensor_0���������
� �
/__inference_sequential_28_layer_call_fn_3492397z
'(01@AIJA�>
7�4
*�'
input_30���������
p

 
� ")�&
unknown����������
/__inference_sequential_28_layer_call_fn_3492422z
'(01@AIJA�>
7�4
*�'
input_30���������
p 

 
� ")�&
unknown����������
%__inference_signature_wrapper_3492488�
'(01@AIJE�B
� 
;�8
6
input_30*�'
input_30���������"Q�N
L
conv2d_transpose_655�2
conv2d_transpose_65���������