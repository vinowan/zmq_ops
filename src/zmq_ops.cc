#include <vector>

#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>

using namespace tensorflow;

namespace avatar {
    REGISTER_OP("ZmqReaderInit")
    .Attr("end_point: string")
    .Attr("hwm: int >= 1 = 100")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Output("resource: resource")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

    REGISTER_OP("ZmqReaderNext")
    .Input("input: resource")
    .Attr("types: list(type) >= 1")
    .Attr("shapes: list(shape) >= 1")
    .Output("output: types")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      std::vector<PartialTensorShape> shape_attrs;
      TF_RETURN_IF_ERROR(c->GetAttr("shapes", &shape_attrs));
      
      std::vector<shape_inference::ShapeHandle> shapes;
      for (size_t i = 0; i < shape_attrs.size(); i++) {
        shape_inference::ShapeHandle s;
        TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(shape_attrs[i], &s));
        shapes.push_back(s);
      }
      TF_RETURN_IF_ERROR(c->set_output("output", shapes));
      return Status::OK();
    });

    REGISTER_OP("ZmqReaderReadable")
    .Input("input: resource")    
    .Output("output: bool")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

    REGISTER_OP("ZmqServerInit")
    .Attr("end_point: string")
    .Attr("hwm: int >= 1 = 100")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Output("resource: resource")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

    REGISTER_OP("ZmqServerRecvAll")
    .Input("input: resource")
    .Attr("types: list(type) >= 1")
    .Attr("shapes: list(shape) >= 1")
    .Attr("min_cnt: int = 1")
    .Attr("max_cnt: int = 16")
    .Output("client_id: string")
    .Output("tensors: types")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->UnknownDim()}));
      
      std::vector<PartialTensorShape> shapes;
      TF_RETURN_IF_ERROR(c->GetAttr("shapes", &shapes));
      
      for (size_t i = 0; i < shapes.size(); i++) {
        shape_inference::ShapeHandle s;
        TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(shapes[i], &s));
        c->set_output(i, s);
      }

      return Status::OK();
    });

    REGISTER_OP("ZmqServerSendAll")
    .Input("input: resource")
    .Input("client_id: string")
    .Input("tensors: types")
    .Attr("types: list(type) >= 1")
    .Output("succ: bool")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    });
};
