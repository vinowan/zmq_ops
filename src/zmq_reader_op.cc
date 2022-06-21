#include <vector>
#include <string>
#include <memory>
#include <sstream>
#include <chrono>

#include <tensorflow/core/lib/core/errors.h>
#include <tensorflow/core/framework/tensor.pb.h>
#include <tensorflow/core/framework/resource_mgr.h>
#include <tensorflow/core/framework/resource_op_kernel.h>
#include <tensorflow/core/platform/thread_annotations.h>

#include "zmq.hpp"
#include "tensor_array.pb.h"

using namespace tensorflow;

namespace avatar {

class ZmqReaderResource: public ResourceBase
{
public:
    ZmqReaderResource(Env* env): env_(env) {}

    Status Init(const std::string& end_point, int hwm)
    {                   
        mutex_lock l(mu_);

        end_point_ = end_point;
        hwm_ = hwm;

        ctx_.reset(new zmq::context_t{1});
        sock_.reset(new zmq::socket_t{*ctx_, zmq::socket_type::pull});

        try {
            sock_->set(zmq::sockopt::linger, 0);
            sock_->set(zmq::sockopt::rcvhwm, hwm_);
            sock_->bind(end_point_.c_str());
        } catch (const zmq::error_t& e) {
            return errors::Internal("failed to init zmq reader resource: ", e.what());
        }

        return Status::OK();
    }

    virtual string DebugString() const {
        std::ostringstream oss;
        oss<<"ZmqReaderResource(end_point: "<<end_point_<<", hwm: "<<hwm_<<")";
        return oss.str();
    }

    Status Next(zmq::message_t* message)
    {  
        {
            // zmq socket is not thread safe
            mutex_lock l(mu_);
            try {
                // block until some data appears
                while (true) {                    
                    zmq::recv_result_t succ = sock_->recv(*message);
                    if (succ && *succ > 0)
                        break;                    
                }
            } catch (zmq::error_t e) {
                return errors::Internal("failed to recv data frome zmq reader resource: ", e.what());
            }
        }

        return Status::OK();
    }

    bool Readable()
    {  
	    bool readable = false;
        
        zmq_pollitem_t item;        
        item.socket = (*sock_.get());
        item.events = ZMQ_POLLIN;

        std::vector<zmq_pollitem_t>  items;
        items.push_back(item);

	    int ret = zmq::poll(items, std::chrono::milliseconds(0)); 
        if (ret > 0) {
            readable = true;
        }

        return readable;
    }

protected:
    mutable mutex mu_;
    
    Env* env_;
    std::string end_point_;
    int hwm_;
    std::unique_ptr<zmq::context_t> ctx_;
    std::unique_ptr<zmq::socket_t> sock_;
};


class ZmqReaderInitOp: public ResourceOpKernel<ZmqReaderResource> 
{
public:
    explicit ZmqReaderInitOp(OpKernelConstruction* context)
        : ResourceOpKernel<ZmqReaderResource>(context) {
        env_ = context->env();

        OP_REQUIRES_OK(context, context->GetAttr("end_point", &end_point_));
        OP_REQUIRES_OK(context, context->GetAttr("hwm", &hwm_));
    }

private:
    Status CreateResource(ZmqReaderResource** resource) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
        *resource = new ZmqReaderResource(env_);
        Status status = (*resource)->Init(end_point_, hwm_);
        return status;
    }

private:
    mutable mutex mu_;
    Env* env_;
    std::string end_point_;
    int hwm_;
};

class ZmqReaderNextOp: public OpKernel
{
public:
    explicit ZmqReaderNextOp(OpKernelConstruction* context)
        : OpKernel(context)
    {
        env_ = context->env();
        
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
        ZmqReaderResource* resource;
        OP_REQUIRES_OK(context, 
            GetResourceFromContext(context, "input", &resource));
        core::ScopedUnref unref(resource);

        zmq::message_t message;
        OP_REQUIRES_OK(context, resource->Next(&message));

        TensorArrayProto tensor_array_proto;
        OP_REQUIRES(
            context, tensor_array_proto.ParseFromArray(message.data(), message.size()),
            errors::Internal("failed to parse tensor array")
        );

        OP_REQUIRES(
            context, 
            tensor_array_proto.tensors_size() == component_types_.size(),
            errors::InvalidArgument("mismatch length: tensor(", tensor_array_proto.tensors_size(), 
                                    "), types(", component_types_.size(), ")")
        );
        
        OpOutputList out_tensors;
        OP_REQUIRES_OK(context, context->output_list("output", &out_tensors));
        
        for(int i = 0; i < tensor_array_proto.tensors_size(); i++) {        
            TensorProto tensor_proto = tensor_array_proto.tensors(i);          
            
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

            Tensor* output = nullptr;
            OP_REQUIRES_OK(context, out_tensors.allocate(i, shape, &output));
            OP_REQUIRES(context, output->FromProto(tensor_proto),
                        errors::Internal("failed to parse from tensor proto")
            );
        }
    }
private:
    mutable mutex mu_;
    Env* env_;

    DataTypeVector component_types_;
    std::vector<PartialTensorShape> component_shapes_;    
};


class ZmqReaderReadableOp: public OpKernel
{
public:
    explicit ZmqReaderReadableOp(OpKernelConstruction* context)
        : OpKernel(context)
    {
        env_ = context->env();        
    }

    void Compute(OpKernelContext* context) override
    {  
        ZmqReaderResource* resource;
        OP_REQUIRES_OK(context, 
            GetResourceFromContext(context, "input", &resource));
        core::ScopedUnref unref(resource);

        bool readable = resource->Readable();
        
        Tensor* output = nullptr;
	    TensorShape output_shape = {};
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

        OP_REQUIRES(
            context, 
            output->CopyFrom(readable ? Tensor(true) : Tensor(false), output_shape), 
            errors::Internal("failed to copy tensor to proto")
        );
    }
private:
    mutable mutex mu_;
    Env* env_;
};


REGISTER_KERNEL_BUILDER(Name("ZmqReaderInit").Device(DEVICE_CPU),
                        ZmqReaderInitOp);
REGISTER_KERNEL_BUILDER(Name("ZmqReaderNext").Device(DEVICE_CPU),
                        ZmqReaderNextOp);
REGISTER_KERNEL_BUILDER(Name("ZmqReaderReadable").Device(DEVICE_CPU),
                        ZmqReaderReadableOp);
};
