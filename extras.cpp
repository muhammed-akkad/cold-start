m.def("load_tensor_from_gpu", &load_tensor_from_gpu, "Load a tensor from a GPU memory pointer");
torch::Tensor load_tensor_from_gpu(uint64_t ptr, std::vector<int64_t> shape, std::vector<int64_t> stride, std::string dtype_str)
{
    torch::ScalarType dtype;

    if (dtype_str == "torch.float32")
    {
        dtype = torch::kFloat32;
    }
    else if (dtype_str == "torch.float64")
    {
        dtype = torch::kFloat64;
    }
    else if (dtype_str == "torch.int32")
    {
        dtype = torch::kInt32;
    }
    else if (dtype_str == "torch.int64")
    {
        dtype = torch::kInt64;
    }
    else
    {
        throw std::runtime_error("Unsupported dtype: " + dtype_str);
    }

    void *data_ptr = reinterpret_cast<void *>(ptr);

    torch::TensorOptions options = torch::TensorOptions().dtype(dtype).device(torch::kCUDA);

    torch::Tensor tensor = torch::from_blob(data_ptr, shape, stride, options);

    return tensor;
}
