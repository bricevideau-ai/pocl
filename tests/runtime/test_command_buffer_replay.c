/* Regression test: repeatedly replaying one command buffer.

   Copyright (c) 2026 pocl developers

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#include <stdio.h>
#include <stdlib.h>

#include "command_buffer_common.h"

#define ELEMENTS 1024
#define REPLAYS 200

/* Nothing here is timing-tuned: see the poll in the replay loop. */

int
main (int _argc, char **_argv)
{
#if defined(cl_khr_command_buffer) && cl_khr_command_buffer == 1
  struct cmdbuf_ext ext;
  cl_platform_id platform;
  CHECK_CL_ERROR (clGetPlatformIDs (1, &platform, NULL));
  cl_device_id device;
  CHECK_CL_ERROR (
      clGetDeviceIDs (platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL));
  int skip = cmdbuf_load_ext (platform, &ext);
  if (skip != 0)
    return skip;

  cl_int error;
  cl_context context = clCreateContext (NULL, 1, &device, NULL, NULL, &error);
  CHECK_CL_ERROR (error);
  cl_command_queue queue
      = clCreateCommandQueueWithProperties (context, device, NULL, &error);
  CHECK_CL_ERROR (error);

  cl_program program;
  cl_kernel kernel;
  CHECK_CL_ERROR (cmdbuf_build_kernel (context, device, &program, &kernel));

  size_t bytes = ELEMENTS * sizeof (cl_int);
  cl_mem a = clCreateBuffer (context, CL_MEM_READ_ONLY, bytes, NULL, &error);
  CHECK_CL_ERROR (error);
  cl_mem b = clCreateBuffer (context, CL_MEM_READ_ONLY, bytes, NULL, &error);
  CHECK_CL_ERROR (error);
  cl_mem res = clCreateBuffer (context, CL_MEM_WRITE_ONLY, bytes, NULL,
                               &error);
  CHECK_CL_ERROR (error);
  CHECK_CL_ERROR (clSetKernelArg (kernel, 0, sizeof (cl_mem), &a));
  CHECK_CL_ERROR (clSetKernelArg (kernel, 1, sizeof (cl_mem), &b));
  CHECK_CL_ERROR (clSetKernelArg (kernel, 2, sizeof (cl_mem), &res));

  /* No CL_COMMAND_BUFFER_SIMULTANEOUS_USE_KHR: the loop below serialises its
     submissions with a blocking wait, so it is not needed. */
  cl_command_buffer_khr cmdbuf
      = ext.clCreateCommandBufferKHR (1, &queue, NULL, &error);
  CHECK_CL_ERROR (error);

  /* More than one command, as a real batched command buffer would have. */
  size_t global = ELEMENTS;
  for (int i = 0; i < 3; ++i)
    CHECK_CL_ERROR (ext.clCommandNDRangeKernelKHR (
        cmdbuf, NULL, NULL, kernel, 1, NULL, &global, NULL, 0, NULL, NULL,
        NULL));
  CHECK_CL_ERROR (ext.clFinalizeCommandBufferKHR (cmdbuf));

  /* Re-enqueue as soon as the previous submission is observably complete.
     cl_khr_command_buffer defines simultaneous use against previous
     submissions "not in the CL_COMPLETE state", so once clGetEventInfo has
     reported CL_COMPLETE this is not simultaneous use and the next enqueue
     must succeed. It did not: the buffer used to be retired only AFTER its
     completion event was published, so an enqueue issued in between was
     rejected CL_INVALID_OPERATION.

     Polling for the status rather than sleeping is what keeps this portable.
     clGetEventInfo takes the event lock and returns event->status, so a
     CL_COMPLETE reading proves the runtime has published the terminal status
     and released that lock -- on any CPU, without a tuned delay. What remains
     is a short race against the retire, so detection is still probabilistic,
     but the window is a fixed span of runtime code rather than a wall-clock
     interval that would need re-tuning per machine. */
  for (int i = 0; i < REPLAYS; ++i)
    {
      cl_event event;
      cl_int err
          = ext.clEnqueueCommandBufferKHR (0, NULL, cmdbuf, 0, NULL, &event);
      if (err != CL_SUCCESS)
        {
          printf ("replay %i failed: %i\n", i, err);
          return EXIT_FAILURE;
        }

      cl_int exec_status = CL_QUEUED;
      do
        {
          CHECK_CL_ERROR (clGetEventInfo (event,
                                          CL_EVENT_COMMAND_EXECUTION_STATUS,
                                          sizeof (exec_status), &exec_status,
                                          NULL));
        }
      while (exec_status > CL_COMPLETE);
      TEST_ASSERT (exec_status == CL_COMPLETE);
      CHECK_CL_ERROR (clReleaseEvent (event));
    }

  CHECK_CL_ERROR (ext.clReleaseCommandBufferKHR (cmdbuf));
  CHECK_CL_ERROR (clReleaseMemObject (a));
  CHECK_CL_ERROR (clReleaseMemObject (b));
  CHECK_CL_ERROR (clReleaseMemObject (res));
  CHECK_CL_ERROR (clReleaseKernel (kernel));
  CHECK_CL_ERROR (clReleaseProgram (program));
  CHECK_CL_ERROR (clReleaseCommandQueue (queue));
  CHECK_CL_ERROR (clReleaseContext (context));
  CHECK_CL_ERROR (clUnloadPlatformCompiler (platform));

  printf ("OK\n");
  return EXIT_SUCCESS;
#else
  return 77;
#endif
}
