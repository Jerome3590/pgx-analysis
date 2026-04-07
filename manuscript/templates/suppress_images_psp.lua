-- suppress_images_psp.lua
-- PSP submission variant of suppress_images.lua.
-- Identical image-suppression logic; captions remain as sibling w:p elements
-- inside the same w:tc, which format_psp_manuscript.py detects and moves to
-- the Figure Legends section at the end of the document.

if FORMAT:match("docx") then

  function Figure(el)
    local src   = ""
    local width = ""
    el:walk({
      Image = function(img)
        if src == "" then
          src   = img.src
          width = img.attributes["width"] or ""
        end
      end
    })
    if src == "" then return el end
    local tag = "[IMAGE:" .. src
    if width ~= "" then tag = tag .. ":" .. width end
    tag = tag .. "]"
    return pandoc.Para({ pandoc.Str(tag) })
  end

  function Image(el)
    local width = el.attributes["width"] or ""
    local tag = "[IMAGE:" .. el.src
    if width ~= "" then tag = tag .. ":" .. width end
    return pandoc.Str(tag .. "]")
  end

end
