#include <array>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <utility>
#include <stdexcept>
#include <iostream>

#include <tensorflow/core/lib/core/errors.h>
#include <tensorflow/core/framework/tensor.pb.h>
#include <tensorflow/core/framework/resource_mgr.h>
#include <tensorflow/core/framework/resource_op_kernel.h>
#include <tensorflow/core/platform/thread_annotations.h>
#include <tensorflow/core/util/batch_util.h>

#include "zmq_addon.hpp"
#include "tensor_array.pb.h"

using namespace tensorflow;

namespace avatar {

using MultiMessages = std::pair<std::string, TensorArrayProto>;

class ZmqServerResource: public ResourceBase
{
public:
    ZmqServerResource(Env* env): env_(env) {}

    Status Init(const std::string& end_point, int hwm)
    {                   
        mutex_lock l(mu_);

        end_point_ = end_point;
        hwm_ = hwm;

        ctx_.reset(new zmq::context_t{1});
        sock_.reset(new zmq::socket_t{*ctx_, zmq::socket_type::router});

        try {
            sock_->set(zmq::sockopt::linger, 0);
            sock_->set(zmq::sockopt::rcvhwm, hwm_);
            sock_->set(zmq::sockopt::sndhwm, hwm_);
            sock_->bind(end_point_.c_str());
        } catch (const zmq::error_t& e) {
            return errors::Internal("failed to init zmq reader resource: ", e.what());
        }

        return Status::OK();
    }

    virtual string DebugString() const {
        std::ostringstream oss;
        oss<<"ZmqServerResource(end_point: "<<end_point_<<", hwm: "<<hwm_<<")";
        return oss.str();
    }

    Status RecvAll(std::vector<MultiMessages>* recv_msgs, int min_cnt, int max_cnt) {
        // zmq socket is not thread safe
        mutex_lock l(mu_);

        int msg_cnt = 0;
        try {
            while(true) {
                std::vector<zmq::message_t> multi_msgs;
                const auto ret = zmq::recv_multipart(*sock_, 
                    back_inserter(multi_msgs), zmq::recv_flags::dontwait);
                if (ret && *ret > 0) {
                    msg_cnt++;
                    
                    // multi_msgs: client_id, delimiter, message, delimiter should be skipped
                    if(*ret != 3)
                        throw std::runtime_error("invalid message format");
                    
                    std::string client_id = multi_msgs[0].to_string();
                    
                    TensorArrayProto tensor_array_proto;
                    if(!tensor_array_proto.ParseFromArray(multi_msgs[2].data(), multi_msgs[2].size()))
                        throw std::runtime_error("failed to parse tensor array proto");

                    recv_msgs->emplace_back(std::move(client_id), std::move(tensor_array_proto));
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }

                if (!ret && msg_cnt >= min_cnt && msg_cnt < max_cnt)
                    break;
            }
        } catch (const std::exception& e) {
            return errors::Internal("failed to recv data from zmq server resource: ", e.what());
        }

        return Status::OK();
    }

    Status SendAll(const std::vector<MultiMessages>& send_msgs) {
        // zmq socket is not thread safe
        mutex_lock l(mu_);
        
        try {
            for(auto& msg: send_msgs) {
                std::vector<zmq::message_t> multipart;
                multipart.emplace_back(zmq::message_t(msg.first));
                multipart.emplace_back(zmq::message_t());
                
                std::string serialized;
                if(!msg.second.SerializeToString(&serialized))
                    throw std::runtime_error("failed to serialize message");
                multipart.emplace_back(zmq::message_t(serialized));

                const auto ret = zmq::send_multipart(*sock_, multipart);
                if(!ret || *ret <= 0)
                    throw std::runtime_error("failed to send multipart message");
            }
        } catch (const std::exception& e) {
            return errors::Internal("failed to send data from zmq server resource: ", e.what());
        }

        return Status::OK();
    }

protected:
    mutable mutex mu_;
    
    Env* env_;
    std::string end_point_;
    int hwm_;
    std::unique_ptr<zmq::context_t> ctx_;
    std::unique_ptr<zmq::socket_t> sock_;
};

class ZmqServerInitOp: public ResourceOpKernel<ZmqServerResource> 
{
public:
    explicit ZmqServerInitOp(OpKernelConstruction* context)
        : ResourceOpKernel<ZmqServerResource>(context) {
        env_ = context->env();

        OP_REQUIRES_OK(context, context->GetAttr("end_point", &end_point_));
        OP_REQUIRES_OK(context, context->GetAttr("hwm", &hwm_));
    }

private:
    Status CreateResource(ZmqServerResource** resource) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
        *resource = new ZmqServerResource(env_);
        Status status = (*resource)->Init(end_point_, hwm_);
        return status;
    }

private:
    mutable mutex mu_;
    Env* env_;
    std::string end_point_;
    int hwm_;
};

class ZmqServerRecvAllOp: public OpKernel
{
public:
    explicit ZmqServerRecvAllOp(OpKernelConstruction* context)
        : OpKernel(context)
    {
        env_ = context->env();
        
        OP_REQUIRES_OK(context, context->GetAttr("min_cnt", &min_cnt_)); 
        OP_REQUIRES_OK(context, context->GetAttr("max_cnt", &max_cnt_));
        OP_REQUIRES(context, 
            min_cnt_ > 0, 
            errors::InvalidArgument("min_cnt should greater than zero")
        );
        OP_REQUIRES(context, 
            min_cnt_ <= max_cnt_, 
            errors::InvalidArgument("min_cnt ", std::to_string(min_cnt_), 
            " is greater than max_cnt ", std::to_string(max_cnt_))
        );

        OP_REQUIRES_OK(context, context->GetAttr("types", &component_types_));
        OP_REQUIRES_OK(context, context->GetAttr("shapes", &component_shapes_));          
        OP_REQUIRES(
            context, 
            component_types_.size()== component_shapes_.size(), 
            errors::InvalidArgument("types and shapes should have same length")
        );
    }

    void Compute(OpKernelContext* context) override
    {
        ZmqServerResource* resource;
        OP_REQUIRES_OK(context, 
            GetResourceFromContext(context, "input", &resource));
        core::ScopedUnref unref(resource);

        std::vector<MultiMessages> recv_msgs;
        OP_REQUIRES_OK(context, resource->RecvAll(&recv_msgs, min_cnt_, max_cnt_));

        size_t num_components = component_types_.size();

        // sanity check
        int64 total_batch_size = 0;
        std::vector<size_t> stride_sizes;
        for (auto& msg: recv_msgs) {
            OP_REQUIRES(
                context, 
                msg.second.tensors_size() == num_components,
                errors::InvalidArgument(
                    "message tensor size ", 
                    std::to_string(msg.second.tensors_size()), 
                    " is different from components size ", 
                    std::to_string(num_components)
                )
            );

            int64 batch_size = -1;
            for (int i = 0; i < msg.second.tensors_size(); i++) {
                const TensorProto& tensor_proto = msg.second.tensors(i);

                DataType dtype = tensor_proto.dtype();
                OP_REQUIRES(
                    context, component_types_[i] == dtype,
                    errors::InvalidArgument("Type mismatch at index ", std::to_string(i),
                                            " between received tensor (", DataTypeString(dtype),
                                            ") and dtype (", DataTypeString(component_types_[i]), ")")
                );

                TensorShape shape(tensor_proto.tensor_shape());
                OP_REQUIRES(
                    context, component_shapes_[i].IsCompatibleWith(shape),
                    errors::InvalidArgument("Shape mismatch at index ", std::to_string(i),
                                            " between received tensor (", shape.DebugString(),
                                            ") and shape (", component_shapes_[i].DebugString(), ")")
                );

                int64 tensor_batch_size = shape.dim_size(0);
                if (batch_size < 0)
                    batch_size = tensor_batch_size;
                OP_REQUIRES(
                    context, 
                    batch_size == tensor_batch_size, 
                    errors::InvalidArgument("batch size mismatch at index ", std::to_string(i), ": ", 
                                            std::to_string(batch_size), " != ", std::to_string(tensor_batch_size))
                );
            }
            stride_sizes.push_back(batch_size);
            total_batch_size += batch_size;
        }

        // allocate output
        Tensor* client_id_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
            "client_id", TensorShape({total_batch_size}), &client_id_tensor));

        OpOutputList output_tensors;
        OP_REQUIRES_OK(context, context->output_list("tensors", &output_tensors));
        OP_REQUIRES(
            context, 
            output_tensors.size() == num_components,
            errors::InvalidArgument("component number mismatch: ", std::to_string(output_tensors.size()), 
                                    " != ", std::to_string(num_components))
        );

        for (size_t i = 0; i < output_tensors.size(); i++) {
            auto dim_sizes = component_shapes_[i].dim_sizes();
            dim_sizes[0] = total_batch_size;
            TensorShape shape;
            OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(dim_sizes, &shape));

            Tensor* unused = nullptr;
            output_tensors.allocate(i, shape, &unused);
        }
        
        // copy data
        size_t dst_offset = 0;
        for (size_t i = 0; i < recv_msgs.size(); i++) {
            auto& msg = recv_msgs[i];

            Tensor client_id(msg.first);
            for(size_t j = 0; j < stride_sizes[i]; j++) {
                OP_REQUIRES_OK(context, batch_util::CopyElementToSlice(
                    client_id, client_id_tensor, dst_offset + j));
            }

            for(size_t j = 0; j < num_components; j++) {
                Tensor tensor;
                OP_REQUIRES(
                    context,
                    tensor.FromProto(msg.second.tensors(j)),
                    errors::Internal("failed to parse proto")
                );
                OP_REQUIRES_OK(context, batch_util::CopyContiguousSlices(
                    tensor, 0, dst_offset, stride_sizes[i], output_tensors[j]));
            }

            dst_offset += stride_sizes[i];
        }
    }

private:
    mutable mutex mu_;
    Env* env_;

    int min_cnt_;
    int max_cnt_;
    
    DataTypeVector component_types_;
    std::vector<PartialTensorShape> component_shapes_;
};

class ZmqServerSendAllOp: public OpKernel
{
public:
    explicit ZmqServerSendAllOp(OpKernelConstruction* context)
        : OpKernel(context)
    {
        env_ = context->env();
        
        OP_REQUIRES_OK(context, context->GetAttr("types", &component_types_));
    }

    void Compute(OpKernelContext* context) override
    {
        ZmqServerResource* resource;
        OP_REQUIRES_OK(context, 
            GetResourceFromContext(context, "input", &resource));
        core::ScopedUnref unref(resource);

        size_t num_components = component_types_.size();
        
        const Tensor* client_id_tensor;
        OP_REQUIRES_OK(context, context->input("client_id", &client_id_tensor));
        const auto& client_ids = client_id_tensor->vec<tstring>();

        OpInputList input_tensors;
        OP_REQUIRES_OK(context, context->input_list("tensors", &input_tensors));

        // sanity check
        int64 batch_size = client_id_tensor->shape().dim_size(0);
        for (size_t i = 0; i < num_components; i++) {
            const Tensor& tensor = input_tensors[i];

            DataType dtype = tensor.dtype();
            OP_REQUIRES(
                context, component_types_[i] == dtype,
                errors::InvalidArgument("Type mismatch at index ", std::to_string(i),
                                        " between received tensor (", DataTypeString(dtype),
                                        ") and dtype (", DataTypeString(component_types_[i]), ")")
            );

            const TensorShape& shape = tensor.shape();
            int64 tensor_batch_size = shape.dim_size(0);
            OP_REQUIRES(context,
                batch_size == tensor_batch_size,
                errors::InvalidArgument("batch size mismatch: ", std::to_string(batch_size), 
                " != ", std::to_string(tensor_batch_size))
            );
        }
        
        std::vector<size_t> stride_sizes;
        const int64 total_batch_size = client_ids.size();
        
        int64 stride_begin_offset = 0;
        int64 stride_end_offset = 0;
        while(stride_end_offset < total_batch_size) {
            const tstring& begin_client_id = client_ids(stride_begin_offset);
            const tstring& end_client_id = client_ids(stride_end_offset);
            if (begin_client_id != end_client_id) {
                stride_sizes.push_back(stride_end_offset - stride_begin_offset);
                stride_begin_offset = stride_end_offset;
            }
            stride_end_offset++;
        }
        stride_sizes.push_back(stride_end_offset - stride_begin_offset);

        // copy data
        size_t offset = 0;
        std::vector<MultiMessages> send_msgs;
        for(size_t stride_size: stride_sizes) {
            std::string client_id = client_ids(offset);

            TensorArrayProto tensor_array_proto;
            for(int i = 0; i < num_components; i++) {
                const Tensor& in_tensor = input_tensors[i];
                
                TensorShape tensor_shape(in_tensor.shape());
                tensor_shape.set_dim(0, stride_size);

                Tensor out_tensor;
                OP_REQUIRES_OK(context, context->allocate_temp(
                    in_tensor.dtype(), tensor_shape, &out_tensor));
                
                batch_util::CopyContiguousSlices(in_tensor, offset, 0, stride_size, &out_tensor);
                TensorProto* tensor_proto = tensor_array_proto.add_tensors();
                out_tensor.AsProtoTensorContent(tensor_proto);
            }
            send_msgs.emplace_back(client_id, tensor_array_proto);
            offset += stride_size;
        }

        Status status = resource->SendAll(send_msgs);

        Tensor* output = nullptr;
	    TensorShape output_shape = {};
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

        OP_REQUIRES(
            context, 
            output->CopyFrom(status.ok() ? Tensor(true) : Tensor(false), output_shape), 
            errors::Internal("failed to copy tensor to proto")
        );
    }

private:
    mutable mutex mu_;
    Env* env_;

    DataTypeVector component_types_;
};

REGISTER_KERNEL_BUILDER(Name("ZmqServerInit").Device(DEVICE_CPU),
                        ZmqServerInitOp);
REGISTER_KERNEL_BUILDER(Name("ZmqServerRecvAll").Device(DEVICE_CPU),
                        ZmqServerRecvAllOp);
REGISTER_KERNEL_BUILDER(Name("ZmqServerSendAll").Device(DEVICE_CPU),
                        ZmqServerSendAllOp);

};