use std::{
    cell::Cell,
    ffi::{CStr, CString},
    iter::FromIterator,
};

use winit::{
    event::{Event, WindowEvent},
    event_loop::{self, ControlFlow, EventLoopWindowTarget},
    window,
};

use ash::{
    extensions::khr::Swapchain,
    prelude::*,
    version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
    vk::{self, SurfaceKHR},
    Entry,
};

use ash::extensions::{ext::DebugUtils, khr::Surface};
use ash_window;

fn event_handler<T>(
    event: Event<T>,
    _target: &EventLoopWindowTarget<T>,
    control_flow: &mut ControlFlow,
    window: &window::Window,
    render: &Render,
) {
    match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            println!("Нажата кнопка закрытия");
            *control_flow = ControlFlow::Exit
        }
        Event::MainEventsCleared => {
            window.request_redraw()
            // Application update code.

            // Queue a RedrawRequested event.
            //
            // You only need to call this if you've determined that you need to redraw, in
            // applications which do not always need to. Applications that redraw continuously
            // can just render here instead.
        }
        Event::RedrawRequested(_) => {
            let sync_object = &render.sync_object;
            let current_frame = render.current_frame.get();
            let wait_fences = [sync_object.in_flight_fences[current_frame]];

            let (image_index, _is_sub_optimal) = unsafe {
                &render
                    .device
                    .wait_for_fences(&wait_fences, true, std::u64::MAX)
                    .expect("Failed to wait for Fence!");

                &render
                    .swapchain
                    .swapchain_loader
                    .acquire_next_image(
                        render.swapchain.swapchain,
                        std::u64::MAX,
                        sync_object.image_available_semaphores[current_frame],
                        vk::Fence::null(),
                    )
                    .expect("Failed to acquire next image.")
            };

            let wait_semaphores = [sync_object.image_available_semaphores[current_frame]];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let signal_semaphores = [sync_object.render_finished_semaphores[current_frame]];

            let submit_infos = [vk::SubmitInfo {
                s_type: vk::StructureType::SUBMIT_INFO,
                wait_semaphore_count: wait_semaphores.len() as u32,
                p_wait_semaphores: wait_semaphores.as_ptr(),
                p_wait_dst_stage_mask: wait_stages.as_ptr(),
                command_buffer_count: 1,
                p_command_buffers: &render.command_buffer[*image_index as usize],
                signal_semaphore_count: signal_semaphores.len() as u32,
                p_signal_semaphores: signal_semaphores.as_ptr(),
                p_next: std::ptr::null(),
            }];

            unsafe {
                render
                    .device
                    .reset_fences(&wait_fences)
                    .expect("Failed to reset Fence!");

                render
                    .device
                    .queue_submit(
                        render.device_queues.graphics_queue,
                        &submit_infos,
                        sync_object.in_flight_fences[current_frame],
                    )
                    .expect("Failed to execute queue submit.");
            }
            render
                .current_frame
                .set((current_frame + 1) % render.framebuffers.len());

            let swapchains = [render.swapchain.swapchain];

            let present_info = vk::PresentInfoKHR {
                s_type: vk::StructureType::PRESENT_INFO_KHR,
                p_next: std::ptr::null(),
                wait_semaphore_count: 1,
                p_wait_semaphores: signal_semaphores.as_ptr(),
                swapchain_count: 1,
                p_swapchains: swapchains.as_ptr(),
                p_image_indices: *&image_index,
                p_results: std::ptr::null_mut(),
            };

            unsafe {
                render
                    .swapchain
                    .swapchain_loader
                    .queue_present(render.device_queues.present_queue, &present_info)
                    .expect("Failed to execute queue present.");
            }
        }
        Event::LoopDestroyed => {
            unsafe {
                render
                    .device
                    .device_wait_idle()
                    .expect("Failed to wait device idle!")
            };
            std::mem::drop(render);
        }
        _ => (),
    }
}

fn main() {
    println!("Привет winit!");

    let event_loop = event_loop::EventLoop::new();

    let window = window::WindowBuilder::new()
        .with_title("Проверка winit на windows 11")
        .build(&event_loop)
        .expect("Не удалось создать window");

    let render = Render::new(&window).unwrap();

    event_loop.run(move |event, target, control_flow| {
        event_handler(event, target, control_flow, &window, &render)
    })
}

struct Render {
    _entry: ash::Entry,
    instance: ash::Instance,
    debug_utils: RDebugUtils,
    surface: RSurface,
    _p_device: vk::PhysicalDevice,
    _query_family_indexes: RQueueFamilyIndex,
    device: ash::Device,
    device_queues: RDeviceQueues,
    swapchain: RSwapChainStuff,
    image_viev: Vec<vk::ImageView>,
    render_pass: vk::RenderPass,
    graphics_pipeline: RGraphicsPipeline,
    framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    command_buffer: Vec<vk::CommandBuffer>,
    sync_object: RSyncObjects,
    current_frame: Cell<usize>,
}

impl Render {
    fn new(window: &window::Window) -> Result<Self, Box<dyn std::error::Error>> {
        let entry = unsafe { Entry::new() }?;

        let mut extensions = ash_window::enumerate_required_extensions(window).unwrap();
        extensions.push(DebugUtils::name());

        let enable_layer_names = vec![CString::new("VK_LAYER_KHRONOS_validation").unwrap()];
        let instance = Self::create_instance(&entry, window, extensions, vec![])?;
        let debug_utils = Self::setup_debug_utils(&entry, &instance)?;
        let surface = Self::create_surface(&entry, &instance, window)?;

        let (p_device, query_family_indexes) =
            Self::pick_physical_device(&instance, &surface, vk::PhysicalDeviceType::DISCRETE_GPU)?;

        //dbg!(unsafe { instance.get_physical_device_properties(p_device) });
        // dbg!(&query_family_indexes);

        let device_extensions = vec![Swapchain::name()];
        let (device, device_queues) = Self::create_logical_device_with_queue_familys(
            &instance,
            &p_device,
            &query_family_indexes,
            device_extensions,
        )?;

        let swapchain = Self::create_swapchain(
            &instance,
            &device,
            p_device,
            &surface,
            &query_family_indexes,
            2,
        )?;

        let image_viev = Self::create_image_views(&device, &swapchain);

        let render_pass = Self::create_render_pass(&device, &swapchain)?;

        let graphics_pipeline = Self::create_graphics_pipeline(&device, &swapchain, &render_pass);

        let framebuffers =
            Self::create_framebuffers(&device, &render_pass, &swapchain, &image_viev);

        let command_pool = Self::create_command_pool(&device, &query_family_indexes)?;

        let command_buffer = Self::create_command_buffer(
            &device,
            &command_pool,
            &framebuffers,
            &render_pass,
            &swapchain,
            &graphics_pipeline,
        )?;

        let sync_object = Self::create_sync_object(&device, &framebuffers);

        Ok(Render {
            _entry: entry,
            instance,
            debug_utils,
            surface,
            _p_device: p_device,
            _query_family_indexes: query_family_indexes,
            device,
            device_queues,
            swapchain,
            image_viev,
            render_pass,
            graphics_pipeline,
            framebuffers,
            command_pool,
            command_buffer,
            sync_object,
            current_frame: Cell::new(0),
        })
    }

    fn create_instance(
        entry: &ash::Entry,
        _window: &window::Window,
        extensions: Vec<&CStr>,
        enable_layers: Vec<CString>,
    ) -> Result<ash::Instance, ash::InstanceError> {
        let app_name = CString::new("App name").unwrap();
        let eng_name = CString::new("Engine name").unwrap();
        let api_vers = vk::make_version(1, 2, 168);
        let app_vers = vk::make_version(0, 0, 1);

        let app_info = vk::ApplicationInfo::builder()
            .api_version(api_vers)
            .application_name(&app_name)
            .engine_name(&eng_name)
            .application_version(app_vers)
            .engine_version(app_vers);

        let extensions_names_raw = extensions
            .iter()
            .map(|ext| ext.as_ptr())
            .collect::<Vec<_>>();

        let enable_layers_raw = enable_layers
            .iter()
            .map(|ext| ext.as_ptr())
            .collect::<Vec<_>>();

        let ins_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&extensions_names_raw)
            .enabled_layer_names(&enable_layers_raw);

        let instance = unsafe { entry.create_instance(&ins_info, None) };

        instance
    }

    fn setup_debug_utils(entry: &ash::Entry, instance: &ash::Instance) -> VkResult<RDebugUtils> {
        let debug_utils_loader = ash::extensions::ext::DebugUtils::new(entry, instance);
        let create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(
                // vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE |
                // vk::DebugUtilsMessageSeverityFlagsEXT::INFO |
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            )
            .pfn_user_callback(Some(vulkan_debug_utils_callback));

        let utils_messenger =
            unsafe { debug_utils_loader.create_debug_utils_messenger(&create_info, None) }?;

        Ok(RDebugUtils {
            debug_utils_loader,
            utils_messenger,
        })
    }

    fn create_surface(
        entry: &Entry,
        instance: &ash::Instance,
        window: &window::Window,
    ) -> VkResult<RSurface> {
        let surface = unsafe { ash_window::create_surface(entry, instance, window, None) }?;
        let surface_loader = Surface::new(entry, instance);

        Ok(RSurface {
            surface,
            surface_loader,
        })
    }

    fn query_swapchain_support(
        p_device: vk::PhysicalDevice,
        surface_stuff: &RSurface,
    ) -> VkResult<RSwapChainSupportDetail> {
        unsafe {
            let capabilities = surface_stuff
                .surface_loader
                .get_physical_device_surface_capabilities(p_device, surface_stuff.surface)?;
            let formats = surface_stuff
                .surface_loader
                .get_physical_device_surface_formats(p_device, surface_stuff.surface)?;
            let present_modes = surface_stuff
                .surface_loader
                .get_physical_device_surface_present_modes(p_device, surface_stuff.surface)?;

            Ok(RSwapChainSupportDetail {
                capabilities,
                formats,
                _present_modes: present_modes,
            })
        }
    }

    fn create_swapchain(
        instance: &ash::Instance,
        device: &ash::Device,
        p_device: vk::PhysicalDevice,
        surface: &RSurface,
        queue_family: &RQueueFamilyIndex,
        image_count: u32,
    ) -> Result<RSwapChainStuff, Box<dyn std::error::Error>> {
        let swapchain_support = Self::query_swapchain_support(p_device, surface)?;

        let surface_format = swapchain_support.formats.first().unwrap();
        let present_mode = vk::PresentModeKHR::FIFO;
        let pre_transform = vk::SurfaceTransformFlagsKHR::IDENTITY;
        let surface_resolution = swapchain_support.capabilities.current_extent;
        let image_count = image_count
            .max(swapchain_support.capabilities.min_image_count)
            .min(swapchain_support.capabilities.max_image_count);

        let queue_family_indices =
            if queue_family.graphyc_family_index != queue_family.present_family_index {
                vec![
                    queue_family.graphyc_family_index,
                    queue_family.present_family_index,
                ]
            } else {
                vec![]
            };

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .queue_family_indices(&queue_family_indices)
            .surface(surface.surface)
            .min_image_count(image_count)
            .image_color_space(surface_format.color_space)
            .image_format(surface_format.format)
            .image_extent(surface_resolution)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(pre_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .image_array_layers(1);

        let swapchain_loader = Swapchain::new(instance, device);
        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None)? };

        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };

        Ok(RSwapChainStuff {
            swapchain_loader,
            swapchain,
            swapchain_format: surface_format.format,
            swapchain_extent: surface_resolution,
            swapchain_images,
        })
    }

    fn pick_physical_device(
        instance: &ash::Instance,
        surface: &RSurface,
        target_type: vk::PhysicalDeviceType,
    ) -> Result<(vk::PhysicalDevice, RQueueFamilyIndex), Box<dyn std::error::Error>> {
        let p_devices = unsafe { instance.enumerate_physical_devices() }?;

        let p_device = p_devices
            .iter()
            .find_map(|&p_device| {
                let p_device_prop = unsafe { instance.get_physical_device_properties(p_device) };
                if p_device_prop.device_type == target_type {
                    Some(p_device)
                } else {
                    None
                }
            })
            .unwrap_or(p_devices[0]);

        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(p_device) };

        let graphyc_family_index = queue_families
            .iter()
            .enumerate()
            .find_map(|(index, &info)| {
                if info.queue_count > 0 && info.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                    Some(index as u32)
                } else {
                    None
                }
            })
            .ok_or("Нет очередей с потдержкой графики!")?;

        let present_family_index = queue_families
            .iter()
            .enumerate()
            .find_map(|(index, &info)| {
                let is_present_support = unsafe {
                    surface.surface_loader.get_physical_device_surface_support(
                        p_device,
                        index as u32,
                        surface.surface,
                    )
                }
                .unwrap();

                if info.queue_count > 0
                    && is_present_support
                    && !info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                {
                    Some(index as u32)
                } else {
                    None
                }
            })
            .unwrap_or(
                queue_families
                    .iter()
                    .enumerate()
                    .find_map(|(index, &info)| {
                        let is_present_support = unsafe {
                            surface.surface_loader.get_physical_device_surface_support(
                                p_device,
                                index as u32,
                                surface.surface,
                            )
                        }
                        .unwrap();

                        if info.queue_count > 0 && is_present_support {
                            Some(index as u32)
                        } else {
                            None
                        }
                    })
                    .ok_or("Нет подходящих очередей с потдержкой прецентации!")?,
            );

        Ok((
            p_device,
            RQueueFamilyIndex {
                graphyc_family_index,
                present_family_index,
            },
        ))
    }

    fn create_logical_device_with_queue_familys(
        instance: &ash::Instance,
        &p_device: &vk::PhysicalDevice,
        query_family_indexes: &RQueueFamilyIndex,
        device_extensions: Vec<&CStr>,
    ) -> VkResult<(ash::Device, RDeviceQueues)> {
        let device_extensions_raw = device_extensions
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();

        let queue_family_indexes = std::collections::BTreeSet::from_iter([
            query_family_indexes.graphyc_family_index,
            query_family_indexes.present_family_index,
        ]);

        let priorities = [1.0];

        let device_queue_create_infos = queue_family_indexes
            .iter()
            .map(|&queue_family_index| {
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(queue_family_index)
                    .queue_priorities(&priorities)
                    .build()
            })
            .collect::<Vec<_>>();

        let features = vk::PhysicalDeviceFeatures::builder()
            .shader_clip_distance(true)
            .fill_mode_non_solid(true);

        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&device_queue_create_infos)
            .enabled_extension_names(&device_extensions_raw)
            .enabled_features(&features);

        let device = unsafe { instance.create_device(p_device, &device_create_info, None) }?;

        let graphics_queue =
            unsafe { device.get_device_queue(query_family_indexes.graphyc_family_index, 0) };
        let present_queue =
            unsafe { device.get_device_queue(query_family_indexes.present_family_index, 0) };

        Ok((
            device,
            RDeviceQueues {
                graphics_queue,
                present_queue,
            },
        ))
    }

    fn create_image_views(device: &ash::Device, swapchain: &RSwapChainStuff) -> Vec<vk::ImageView> {
        let surface_format = swapchain.swapchain_format;
        swapchain
            .swapchain_images
            .iter()
            .map(|&image| {
                let imageview_create_info = vk::ImageViewCreateInfo::builder()
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(surface_format)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::IDENTITY,
                        g: vk::ComponentSwizzle::IDENTITY,
                        b: vk::ComponentSwizzle::IDENTITY,
                        a: vk::ComponentSwizzle::IDENTITY,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image(image);
                unsafe {
                    device
                        .create_image_view(&imageview_create_info, None)
                        .expect("Failed to create Image View!")
                }
            })
            .collect::<Vec<_>>()
    }

    fn create_render_pass(
        device: &ash::Device,
        swapchain: &RSwapChainStuff,
    ) -> VkResult<vk::RenderPass> {
        // Прежде чем мы сможем завершить создание конвейера, нам нужно сообщить Vulkan о прикреплениях фреймбуфера,
        // которые будут использоваться при рендеринге. Нам нужно указать, сколько будет буферов цвета и глубины,
        // сколько сэмплов использовать для каждого из них и как их содержимое должно обрабатываться во время операций рендеринга.
        // Вся эта информация заключена в объект прохода рендеринга

        // В нашем случае у нас будет только одно прикрепление буфера цвета,
        // представленное одним из изображений из цепочки подкачки
        let color_attachment = vk::AttachmentDescription::builder()
            .format(swapchain.swapchain_format)
            // Прикрепленный цвета должны соответствовать формату цепь изображений свопа,
            // и мы не делаем ничего с мультисэмплинг еще, так что мы будем придерживаться 1 образца.
            .samples(vk::SampleCountFlags::TYPE_1)
            // У нас есть следующие варианты load_op:
            //      vk::AttachmentLoadOp::LOAD:         Сохранить существующее содержимое вложения
            //      vk::AttachmentLoadOp::CLEAR:        Очистить значения до константы в начале
            //      vk::AttachmentLoadOp::DONT_CARE:    Существующее содержимое не определено; мы не заботимся о них
            .load_op(vk::AttachmentLoadOp::CLEAR)
            //Есть только две возможности store_op:
            //      vk::AttachmentStoreOp::STORE:       Обработанное содержимое будет сохранено в памяти и может быть прочитано позже.
            //      vk::AttachmentStoreOp::DONT_CARE:   Содержимое фреймбуфера будет неопределенным после операции рендеринга.
            .store_op(vk::AttachmentStoreOp::STORE)
            // stencil_load_op/ stencil_store Opприменимы к данным трафарета.
            // Наше приложение ничего не делает с буфером трафарета,
            // поэтому результаты загрузки и сохранения не имеют значения.
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            // Текстуры и фреймбуферы в Vulkan представлены VkImageобъектами с определенным форматом пикселей,
            // однако расположение пикселей в памяти может меняться в зависимости от того, что вы пытаетесь сделать с изображением.
            // Вот некоторые из наиболее распространенных макетов:
            //      vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL: Изображения используются как цветные вложения.
            //      vk::ImageLayout::PRESENT_SRC_KHR:          Изображения, которые будут представлены в цепочке обмена
            //      vk::ImageLayout::TRANSFER_DST_OPTIMAL:     Изображения, которые будут использоваться в качестве места назначения для операции копирования из памяти
            //      vk::ImageLayout::UNDEFINED:                Предостережение этого специального значения заключается в том,
            // что не гарантируется сохранение содержимого изображения, но это не имеет значения, поскольку мы собираемся очистить это все равно.
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .build();

        // Один проход рендеринга может состоять из нескольких подпроходов.
        // Подпроходы - это последующие операции рендеринга, которые зависят от содержимого кадровых буферов на предыдущих проходах,
        // например, последовательность эффектов постобработки, которые применяются один за другим.
        // Если вы сгруппируете эти операции рендеринга в один проход рендеринга, то Vulkan сможет изменить порядок операций и
        // сохранить полосу пропускания памяти для, возможно, лучшей производительности.
        // Однако для нашего самого первого треугольника мы будем придерживаться одного подпрохода

        let color_attachment_ref = vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::builder()
            // Vulkan может также поддерживать подпроходы вычислений в будущем, поэтому мы должны четко указать, что это подпроходы графики
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            // На индекс вложения в этом массиве напрямую ссылается фрагментный шейдер
            // с помощью layout(location = 0) out vec4 outColorдирективы!
            //Подпроходом могут быть ссылки на следующие другие типы вложений:
            //      p_input_attachments:        Вложения, считываемые из шейдера.
            //      p_resolve_attachments:      Вложения, используемые для вложений цветов с множественной выборкой
            //      p_depthStencil_attachment:  Приложение для данных глубины и трафарета
            //      p_preserve_attachments:     Вложения, которые не используются этим подпроходом, но для которых необходимо сохранить данные.
            .color_attachments(std::slice::from_ref(&color_attachment_ref));

        let render_pass_attachments = std::slice::from_ref(&color_attachment);

        // Tеперь, когда были описаны вложение и базовый подпроход, ссылающийся на него, мы можем создать сам проход рендеринга.
        let renderpass_create_info = vk::RenderPassCreateInfo::builder()
            .attachments(render_pass_attachments)
            .subpasses(std::slice::from_ref(&subpass));

        Ok(unsafe { device.create_render_pass(&renderpass_create_info, None)? })
    }

    fn create_graphics_pipeline(
        device: &ash::Device,
        swapchain: &RSwapChainStuff,
        &render_pass: &vk::RenderPass,
    ) -> RGraphicsPipeline {
        let vert_shader_module = {
            let vert_shader_code = include_bytes!("spv/vert.spv");

            let shader_module_create_info = vk::ShaderModuleCreateInfo {
                code_size: vert_shader_code.len(),
                p_code: vert_shader_code.as_ptr() as *const u32,
                ..Default::default()
            };

            unsafe {
                device
                    .create_shader_module(&shader_module_create_info, None)
                    .expect("Failed to create Shader Module from vert.spv!")
            }
        };

        let frag_shader_module = {
            let frag_shader_code = include_bytes!("spv/frag.spv");

            let shader_module_create_info = vk::ShaderModuleCreateInfo {
                code_size: frag_shader_code.len(),
                p_code: frag_shader_code.as_ptr() as *const u32,
                ..Default::default()
            };

            unsafe {
                device
                    .create_shader_module(&shader_module_create_info, None)
                    .expect("Failed to create Shader Module from frag.spv!")
            }
        };

        let main_function_name = CString::new("main").unwrap();

        let shader_stages = vec![
            vk::PipelineShaderStageCreateInfo::builder()
                .module(vert_shader_module)
                .name(&main_function_name)
                .stage(vk::ShaderStageFlags::VERTEX)
                .build(),
            vk::PipelineShaderStageCreateInfo::builder()
                .module(frag_shader_module)
                .name(&main_function_name)
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .build(),
        ];

        let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo::default();
        let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);
        let viewports = vec![vk::Viewport::builder()
            .x(0.)
            .y(0.)
            .width(swapchain.swapchain_extent.width as f32)
            .height(swapchain.swapchain_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0)
            .build()];

        let scissors = vec![vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: swapchain.swapchain_extent,
        }];

        let viewport_state_create_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewports)
            .scissors(&scissors);

        let rasterization_statue_create_info = vk::PipelineRasterizationStateCreateInfo::builder()
            // Если depth_clamp_enable(true), то фрагменты, которые находятся за пределами ближней и дальней плоскостей,
            // прижимаются к ним, а не отбрасываются. Это полезно в некоторых особых случаях, например, в картах теней.
            // Для этого требуется включить функцию графического процессора.
            .depth_clamp_enable(false)
            // Если rasterizer_discard_enable(true), то геометрия никогда не проходит через этап растеризации.
            // Это в основном отключает любой вывод в буфер кадра
            .rasterizer_discard_enable(false)
            // polygon_mode oпределяет , как фрагменты генерируются для геометрии. Доступны следующие режимы:
            //     polygvk::PolygonMode::FILL: заполнить область многоугольника фрагментами
            //     polygvk::PolygonMode::LINE: края многоугольника рисуются как линии
            //     polygvk::PolygonMode::POINT: вершины многоугольника рисуются как точки
            // Для использования любого режима, кроме заливки, необходимо включить функцию графического процессора.
            .polygon_mode(vk::PolygonMode::FILL)
            // Элемент line_width прост, он описывает толщину линий по количеству фрагментов.
            // Максимальная поддерживаемая ширина линии зависит от оборудования и любой линии, более толстой,
            // чем 1.0f требуется  включение функции wide_lines графического процессора.
            .line_width(1.0)
            // Вы можете отключить отбраковку, отсечь передние грани, отсечь задние грани или и то, и другое
            .cull_mode(vk::CullModeFlags::BACK)
            // Указываем как определять какие стороны треугольника передние:  по часовой, или против
            .front_face(vk::FrontFace::CLOCKWISE)
            // Растеризатор может изменять значения глубины, добавляя постоянное значение или смещая их в зависимости от наклона фрагмента.
            // Иногда это используется для отображения теней
            .depth_bias_clamp(0.0)
            .depth_bias_constant_factor(0.0)
            .depth_bias_enable(false)
            .depth_bias_slope_factor(0.0);
        let multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let stencil_state = vk::StencilOpState::default();

        let depth_state_create_info = vk::PipelineDepthStencilStateCreateInfo::builder()
            .front(stencil_state)
            .back(stencil_state)
            .max_depth_bounds(1.0)
            .min_depth_bounds(0.0);

        let color_blend_attachment_states = vec![vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::all())
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD)
            .build()];
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&color_blend_attachment_states)
            .blend_constants([0.0, 0.0, 0.0, 0.0]);

        let pipeline_layout = {
            // Вы можете использовать uniform значения в шейдерах, которые являются глобальными переменными, аналогичными динамическим переменным состояния,
            // которые можно изменять во время рисования, чтобы изменить поведение ваших шейдеров без необходимости их воссоздания.
            // Обычно они используются для передачи матрицы преобразования в вершинный шейдер или для создания сэмплеров текстуры во фрагментном шейдере.
            // Эти единые значения необходимо указать во время создания конвейера путем создания VkPipelineLayout объекта.
            // Несмотря на то, что мы не будем использовать их до следующей главы, нам все равно необходимо создать пустой макет конвейера.

            let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default();
            let pipeline_layout = unsafe {
                device
                    .create_pipeline_layout(&pipeline_layout_create_info, None)
                    .expect("Failed to create pipeline layout!")
            };
            pipeline_layout
        };

        let graphic_pipeline_create_infos = [vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_state_create_info)
            .input_assembly_state(&vertex_input_assembly_state_info)
            .viewport_state(&viewport_state_create_info)
            .rasterization_state(&rasterization_statue_create_info)
            .multisample_state(&multisample_state_create_info)
            .depth_stencil_state(&depth_state_create_info)
            .color_blend_state(&color_blend_state)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .build()];

        let graphics_pipelines = unsafe {
            device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &graphic_pipeline_create_infos,
                    None,
                )
                .expect("Failed to create Graphics Pipeline!.")
        };

        unsafe {
            device.destroy_shader_module(vert_shader_module, None);
            device.destroy_shader_module(frag_shader_module, None);
        }

        RGraphicsPipeline {
            graphics_pipeline: graphics_pipelines[0],
            pipeline_layout,
        }
    }

    fn create_framebuffers(
        device: &ash::Device,
        &render_pass: &vk::RenderPass,
        swapchain: &RSwapChainStuff,
        image_viev: &Vec<vk::ImageView>,
    ) -> Vec<vk::Framebuffer> {
        image_viev
            .iter()
            .map(|image_view| {
                let attachments = std::slice::from_ref(image_view);

                let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(render_pass)
                    .attachments(attachments)
                    .width(swapchain.swapchain_extent.width)
                    .height(swapchain.swapchain_extent.height)
                    .layers(1);

                unsafe {
                    device
                        .create_framebuffer(&framebuffer_create_info, None)
                        .expect("Failed to create Framebuffer!")
                }
            })
            .collect::<Vec<_>>()
    }

    fn create_command_pool(
        device: &ash::Device,
        query_family_indexes: &RQueueFamilyIndex,
    ) -> VkResult<vk::CommandPool> {
        let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(query_family_indexes.graphyc_family_index)
            // Есть два возможных флага для пулов команд:
            //
            //      VK_COMMAND_POOL_CREATE_TRANSIENT_BIT:            Подсказка, что командные буферы очень часто
            // перезаписываются новыми командами (может изменить поведение выделения памяти)
            //
            //      VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: Разрешить перезапись буферов команд по отдельности,
            // без этого флага все они должны быть сброшены вместе
            .flags(vk::CommandPoolCreateFlags::empty());

        Ok(unsafe { device.create_command_pool(&command_pool_create_info, None) }?)
    }

    fn create_command_buffer(
        device: &ash::Device,
        &command_pool: &vk::CommandPool,
        framebuffers: &Vec<vk::Framebuffer>,
        &render_pass: &vk::RenderPass,
        swapchain: &RSwapChainStuff,
        graphics_pipeline: &RGraphicsPipeline,
    ) -> VkResult<Vec<vk::CommandBuffer>> {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .command_buffer_count(framebuffers.len() as u32)
            //В levelопределяет параметр , если выделенные командные буфера являются первичными или вторичными буферами команд.
            //      vk::CommandBufferLevel::PRIMARY:    Может быть отправлен в очередь для выполнения, но не может быть вызван из других буферов команд.
            //      vk::CommandBufferLevel::SECONDARY:  Не может быть отправлено напрямую, но может быть вызвано из первичных командных буферов.
            .level(vk::CommandBufferLevel::PRIMARY);

        let command_buffers =
            unsafe { device.allocate_command_buffers(&command_buffer_allocate_info) }?;

        for (i, &command_buffer) in command_buffers.iter().enumerate() {
            // Мы начинаем запись командного буфера с вызова begin_command_buffer небольшой  vk::CommandBufferBeginInfo структурой
            // в качестве аргумента, который указывает некоторые детали использования этого конкретного командного буфера.
            let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
                // В flagsопределяет параметр , как мы будем использовать буфер команд. Доступны следующие значения:
                //      vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT:        Командный буфер будет перезаписан сразу после его выполнения.
                //      vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE:   Это дополнительный буфер команд, который будет полностью находиться в пределах одного прохода рендеринга.
                //      vk::CommandBufferUsageFlags::SIMULTANEOUS_USE:       Командный буфер можно повторно отправить, пока он уже ожидает выполнения.
                .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);

            unsafe {
                device
                    .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                    .expect("Failed to begin recording Command Buffer at beginning!");
            }

            let clear_values = [vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            }];

            // Рисование начинается с начала прохода рендеринга с cmd_begin_render_pass.
            // Этап рендеринга настраивается с использованием некоторых параметров в RenderPassBeginInfo.

            let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                .render_pass(render_pass)
                .framebuffer(framebuffers[i])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: swapchain.swapchain_extent,
                })
                .clear_values(&clear_values);

            unsafe {
                device.cmd_begin_render_pass(
                    command_buffer,
                    &render_pass_begin_info,
                    vk::SubpassContents::INLINE,
                );
                device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    graphics_pipeline.graphics_pipeline,
                );
                device.cmd_draw(command_buffer, 3, 1, 0, 0);

                device.cmd_end_render_pass(command_buffer);

                device
                    .end_command_buffer(command_buffer)
                    .expect("Failed to record Command Buffer at Ending!");
            }
        }

        Ok(command_buffers)
    }

    fn create_sync_object(
        device: &ash::Device,
        framebuffers: &Vec<vk::Framebuffer>,
    ) -> RSyncObjects {
        let mut sync_objects = RSyncObjects::default();
        let semaphore_create_info = vk::SemaphoreCreateInfo::default();
        let fence_create_info = vk::FenceCreateInfo::builder()
            .flags(vk::FenceCreateFlags::SIGNALED)
            .build();
        for _ in 0..framebuffers.len() {
            unsafe {
                let image_available_semaphore = device
                    .create_semaphore(&semaphore_create_info, None)
                    .expect("Failed to create Semaphore Object!");
                let render_finished_semaphore = device
                    .create_semaphore(&semaphore_create_info, None)
                    .expect("Failed to create Semaphore Object!");
                let inflight_fence = device
                    .create_fence(&fence_create_info, None)
                    .expect("Failed to create Fence Object!");

                sync_objects
                    .image_available_semaphores
                    .push(image_available_semaphore);
                sync_objects
                    .render_finished_semaphores
                    .push(render_finished_semaphore);
                sync_objects.in_flight_fences.push(inflight_fence);
            }
        }
        sync_objects
    }
}

impl Drop for Render {
    fn drop(&mut self) {
        unsafe {
            self.sync_object
                .image_available_semaphores
                .iter()
                .for_each(|&semaphore| self.device.destroy_semaphore(semaphore, None));
            self.sync_object
                .render_finished_semaphores
                .iter()
                .for_each(|&semaphore| self.device.destroy_semaphore(semaphore, None));
            self.sync_object
                .in_flight_fences
                .iter()
                .for_each(|&fence| self.device.destroy_fence(fence, None));

            self.device.destroy_command_pool(self.command_pool, None);
            self.framebuffers
                .iter()
                .for_each(|&framebuffer| self.device.destroy_framebuffer(framebuffer, None));
            self.device
                .destroy_pipeline_layout(self.graphics_pipeline.pipeline_layout, None);
            self.device
                .destroy_pipeline(self.graphics_pipeline.graphics_pipeline, None);
            self.device.destroy_render_pass(self.render_pass, None);
            self.image_viev
                .iter()
                .for_each(|&image_view| self.device.destroy_image_view(image_view, None));
            self.swapchain
                .swapchain_loader
                .destroy_swapchain(self.swapchain.swapchain, None);
            self.device.destroy_device(None);
            self.surface
                .surface_loader
                .destroy_surface(self.surface.surface, None);
            self.debug_utils
                .debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_utils.utils_messenger, None);
            self.instance.destroy_instance(None);
        }
    }
}

struct RDebugUtils {
    debug_utils_loader: ash::extensions::ext::DebugUtils,
    utils_messenger: vk::DebugUtilsMessengerEXT,
}

struct RSurface {
    surface: SurfaceKHR,
    surface_loader: Surface,
}

struct RSwapChainSupportDetail {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    _present_modes: Vec<vk::PresentModeKHR>,
}
struct RSwapChainStuff {
    swapchain_loader: ash::extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
}

#[derive(Debug)]
struct RQueueFamilyIndex {
    graphyc_family_index: u32,
    present_family_index: u32,
}

struct RDeviceQueues {
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
}

struct RGraphicsPipeline {
    graphics_pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
}

#[derive(Default)]
struct RSyncObjects {
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
}

unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let severity = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "[Verbose]",
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "[Warning]",
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "[Error]",
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "[Info]",
        _ => "[Unknown]",
    };
    let types = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[General]",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[Performance]",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[Validation]",
        _ => "[Unknown]",
    };
    let message = std::ffi::CStr::from_ptr((*p_callback_data).p_message);
    println!("[Debug]{}{}{:?}", severity, types, message);

    vk::FALSE
}
