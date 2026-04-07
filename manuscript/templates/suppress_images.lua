-- suppress_images.lua
-- Quarto/Pandoc Lua filter (Pandoc 3.x): replaces Figure blocks and inline Images
-- with plain-text placeholders so insert_docx_images.py can re-insert them.
--
-- Placeholder format:  [IMAGE:relative/path/to/figure.png]

if FORMAT:match("docx") then

  -- Pandoc 3.x: labeled figures become Figure blocks
  function Figure(el)
    local src = ""
    local width = ""
    -- Walk figure content to find the first Image src and width
    el:walk({
      Image = function(img)
        if src == "" then
          src = img.src
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

  -- Inline images (no label / bare ![](path))
  function Image(el)
    local width = el.attributes["width"] or ""
    local tag = "[IMAGE:" .. el.src
    if width ~= "" then tag = tag .. ":" .. width end
    return pandoc.Str(tag .. "]")
  end

end
